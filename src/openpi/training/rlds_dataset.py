"""Generic base RLDS dataset loader.

Extracts common RLDS loading logic (dataset building, action chunking, flattening,
shuffling, batching) from the DROID-specific implementation into a reusable base class.
Subclasses override trajectory_transforms() and RL hooks to customize dataset-specific
processing.

Supports two modes:
- Training mode (default): Infinite streaming with flattening, shuffling, and batching.
- Trajectory mode (return_trajectories=True): Finite dataset yielding full trajectories
  for validation, with frame transforms applied per-timestep via traj_map.
"""

from collections.abc import Sequence
import dataclasses
import logging
from typing import Any

import openpi.training.counterfactual_action_store as counterfactual_action_store
import openpi.training.latent_store as latent_store


@dataclasses.dataclass
class RLDSDataset:
    """Specification for a single RLDS dataset to load."""

    name: str
    version: str
    weight: float
    filter_dict_path: str | None = None


class BaseRldsDataset:
    """Generic RLDS dataset loader with customization hooks.

    Handles the common RLDS pipeline: loading via dlimp/tfds, repeating for infinite
    streaming, action chunking, flattening trajectories to frames, combining multiple
    datasets with weighted sampling, shuffle buffering, and batching.

    Supports multi-host training by sharding the dataset across JAX processes.

    Subclasses override:
        trajectory_transforms(traj, dataset_cfg): Per-trajectory processing after loading.
        _get_per_step_done_flags(raw_traj, mapped_traj, per_step_rewards): Optional RL done semantics.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDataset],
        *,
        split: str = "train",
        shuffle: bool = True,
        shuffle_seed: int = 42,
        action_chunk_size: int = 16,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        critic_mode: bool = False,
        discount: float = 0.99,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        image_obs_keys: Sequence[str] = (),
        include_images: bool = True,
        return_trajectories: bool = False,
        max_trajectories: int | None = None,
        latent_store_dir: str | None = None,
        latent_views: Sequence[latent_store.LatentViewConfig] = (),
        max_num_demos: int | None = None,
        counterfactual_action_store_dir: str | None = None,
        repeat_dataset: bool = True,
    ):
        if max_num_demos is not None and max_num_demos <= 0:
            raise ValueError(f"max_num_demos must be a positive integer, got {max_num_demos}")
        self._max_num_demos = max_num_demos

        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import jax
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")

        assert sum(dataset.weight for dataset in datasets) == 1.0, "Dataset weights must sum to 1.0"

        self._num_parallel_calls = num_parallel_calls
        self._num_parallel_reads = num_parallel_reads
        self._action_chunk_size = action_chunk_size
        self._critic_mode = critic_mode
        self._discount = discount
        self._reward_scale = reward_scale
        self._reward_bias = reward_bias
        self._include_images = include_images
        self._image_obs_keys = tuple(image_obs_keys) if include_images else ()
        self._return_trajectories = return_trajectories
        self._max_trajectories = max_trajectories
        self._latent_store_dir = latent_store_dir
        self._latent_views = tuple(latent_views)
        self._counterfactual_action_store_dir = counterfactual_action_store_dir
        self._shuffle = shuffle
        self._shuffle_seed = shuffle_seed

        # Load latent store manifest if configured
        self._latent_manifest: latent_store.LatentStoreManifest | None = None
        if latent_store_dir is not None and latent_views:
            self._latent_manifest = latent_store.load_manifest(latent_store_dir)
            logging.info(
                f"Loaded latent store manifest: {self._latent_manifest.source_dataset_name} "
                f"v{self._latent_manifest.source_dataset_version}, "
                f"{len(self._latent_manifest.image_keys)} cameras, "
                f"temporal_compression={self._latent_manifest.temporal_compression}"
            )

        # Load counterfactual action store manifest if configured
        self._ca_manifest: counterfactual_action_store.CounterfactualActionStoreManifest | None = None
        if counterfactual_action_store_dir is not None:
            self._ca_manifest = counterfactual_action_store.load_manifest(counterfactual_action_store_dir)
            logging.info(
                f"Loaded counterfactual action store manifest: {self._ca_manifest.source_dataset_name} "
                f"v{self._ca_manifest.source_dataset_version}, "
                f"num_samples={self._ca_manifest.num_samples}, "
                f"action_horizon={self._ca_manifest.action_horizon}, "
                f"action_dim={self._ca_manifest.action_dim}"
            )

        # Cache short_to_full_key mapping for latent view lookups (computed once, not per-trajectory)
        self._short_to_full_key: dict[str, str] = {}
        if self._latent_manifest is not None:
            for full_key in self._latent_manifest.image_keys:
                short_key = full_key.split("/")[-1]
                if short_key in self._short_to_full_key:
                    raise ValueError(
                        f"Duplicate short key '{short_key}' from image keys "
                        f"'{self._short_to_full_key[short_key]}' and '{full_key}'. "
                        f"Use explicit image_keys in LatentViewConfig to disambiguate."
                    )
                self._short_to_full_key[short_key] = full_key

        # For multi-host training, each process loads its own shard
        process_count = jax.process_count()
        process_index = jax.process_index()

        if process_count > 1 and not return_trajectories:
            assert batch_size % process_count == 0, (
                f"batch_size ({batch_size}) must be divisible by process_count ({process_count})"
            )
            local_batch_size = batch_size // process_count
            logging.info(
                f"Multi-host RLDS loading: process {process_index}/{process_count}, local_batch_size={local_batch_size}"
            )
        else:
            local_batch_size = batch_size

        def prepare_single_dataset(dataset_cfg: RLDSDataset, *, for_trajectories: bool):
            """Prepare a single dataset for either training or trajectory mode.

            Args:
                dataset_cfg: Dataset configuration.
                for_trajectories: If True, prepare for trajectory mode (finite, yields trajectories).
                                  If False, prepare for training (infinite, flattened frames).
            """
            builder = tfds.builder(dataset_cfg.name, data_dir=data_dir, version=dataset_cfg.version)

            if for_trajectories:
                # Use strided sampling for validation to get diverse trajectories across the dataset.
                # TFDS doesn't support stride syntax, so we build a union of individual indices.
                if max_trajectories is not None:
                    num_episodes = builder.info.splits[split].num_examples
                    stride = max(1, num_episodes // max_trajectories)
                    indices = list(range(0, num_episodes, stride))[:max_trajectories]
                    split_to_use = "+".join(f"{split}[{i}:{i + 1}]" for i in indices)
                else:
                    split_to_use = split
                do_shuffle = False
            else:
                # Apply max_num_demos limit if specified (uses TFDS absolute-count slicing)
                base_split = f"{split}[:{self._max_num_demos}]" if self._max_num_demos is not None else split
                split_to_use = (
                    tfds.split_for_jax_process(base_split, process_index=process_index, process_count=process_count)
                    if process_count > 1
                    else base_split
                )
                logging.info(f"  Dataset {dataset_cfg.name}: using split {split_to_use!r}")
                do_shuffle = shuffle

            read_config_kwargs = {"shuffle_seed": shuffle_seed} if do_shuffle else None

            dataset = dl.DLataset.from_rlds(
                builder,
                split=split_to_use,
                shuffle=do_shuffle,
                num_parallel_reads=num_parallel_reads,
                read_config_kwargs=read_config_kwargs,
            )

            # Join with latent store if configured (before repeat to maintain episode alignment)
            if self._latent_store_dir is not None and self._latent_views and not for_trajectories:
                dataset = self._join_latent_store(
                    dataset,
                    dataset_cfg,
                    split,
                    process_index if process_count > 1 else 0,
                    max(1, process_count),
                )

            # Join with counterfactual action store if configured
            if self._counterfactual_action_store_dir is not None and not for_trajectories:
                dataset = self._join_counterfactual_action_store(
                    dataset,
                    dataset_cfg,
                    split,
                    process_index if process_count > 1 else 0,
                    max(1, process_count),
                )

            if not for_trajectories and repeat_dataset:
                dataset = dataset.repeat()

            dataset = dataset.traj_map(lambda traj: self._prepare_trajectory(traj, dataset_cfg), num_parallel_calls)
            dataset = dataset.traj_map(lambda traj: self._chunk_actions(traj, action_chunk_size), num_parallel_calls)

            if for_trajectories:
                # Apply frame transforms to each step within the trajectory
                dataset = dataset.traj_map(
                    lambda traj: self._apply_frame_transforms_to_trajectory(traj), num_parallel_calls
                )
            else:
                dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)
                dataset = dataset.frame_map(lambda frame: self.frame_transforms(frame), num_parallel_calls)
                dataset = dataset.filter(lambda frame: self.frame_filter(frame))
            return dataset

        logging.info(f"Preparing {len(datasets)} datasets (split={split}, trajectory_mode={return_trajectories})...")
        logging.info("-" * 50)
        for dataset in datasets:
            logging.info(f"    {dataset.name}:{dataset.version} with weight {dataset.weight:.2f}")
        logging.info("-" * 50)

        if return_trajectories:
            # Trajectory mode: dataset yields full trajectories with frame transforms applied
            all_datasets = [prepare_single_dataset(dataset_cfg, for_trajectories=True) for dataset_cfg in datasets]
            # Concatenate datasets (for trajectory mode we don't sample, just concatenate)
            if len(all_datasets) == 1:
                final_dataset = all_datasets[0]
            else:
                final_dataset = all_datasets[0]
                for ds in all_datasets[1:]:
                    final_dataset = final_dataset.concatenate(ds)
            self.dataset = final_dataset
            logging.info("Dataset prepared in trajectory mode (lazy evaluation)")
        else:
            # Training mode: streaming with flattening
            all_datasets = [prepare_single_dataset(dataset_cfg, for_trajectories=False) for dataset_cfg in datasets]
            weights = [dataset.weight for dataset in datasets]

            final_dataset = dl.DLataset.sample_from_datasets(all_datasets, weights=weights)
            if shuffle:
                final_dataset = final_dataset.shuffle(shuffle_buffer_size)
            final_dataset = final_dataset.batch(local_batch_size)
            final_dataset = final_dataset.with_ram_budget(1)
            self.dataset = final_dataset

        self.batch_size = batch_size  # Global batch size
        self.local_batch_size = local_batch_size
        self.shuffle = shuffle

    def _join_latent_store(
        self,
        dataset,
        dataset_cfg: RLDSDataset,
        split: str,
        process_index: int,
        process_count: int,
    ):
        """Join latent store episodes with RLDS episodes at trajectory level.

        Uses TFDS builder to load the latent store with IDENTICAL settings as the RLDS
        dataset (shuffle_seed, process sharding via split_for_jax_process). This ensures
        both datasets yield episodes in the same order.

        CRITICAL: Both datasets MUST use deterministic ordering for the zip to work.
        We apply tf.data.Options with deterministic=True to override dlimp's default
        non-deterministic mode. This ensures file interleaving is consistent.

        Args:
            dataset: The RLDS DLataset to join with.
            dataset_cfg: Dataset configuration (for validation).
            split: Split name (e.g., "train").
            process_index: This process's index for sharding.
            process_count: Total number of processes.

        Returns:
            Joined dataset where each trajectory has latent data attached.
        """
        import tensorflow as tf
        import tensorflow_datasets as tfds

        if self._latent_manifest is None:
            return dataset

        # Validate manifest matches the dataset
        latent_store.validate_manifest_against_rlds(
            self._latent_manifest,
            rlds_data_dir="",  # Skip path check
            dataset_name=dataset_cfg.name,
            dataset_version=dataset_cfg.version,
        )

        # Load latent store as TFDS builder (ensures identical loading behavior)
        latent_builder = latent_store.get_latent_store_builder(self._latent_store_dir)

        # Use SAME process sharding as RLDS (tfds.split_for_jax_process)
        split_to_use = (
            tfds.split_for_jax_process(split, process_index=process_index, process_count=process_count)
            if process_count > 1
            else split
        )

        # CRITICAL: Use SAME read_config as RLDS (shuffle_seed + parallel reads for file ordering)
        # Both must use identical interleave settings for consistent episode ordering.
        read_config = tfds.ReadConfig(
            skip_prefetch=True,  # Match dlimp's from_rlds
            shuffle_seed=self._shuffle_seed if self._shuffle else None,
            num_parallel_calls_for_interleave_files=self._num_parallel_reads,
            interleave_cycle_length=self._num_parallel_reads,
        )

        # Load latent store with identical settings to RLDS
        latent_ds = latent_builder.as_dataset(
            split=split_to_use,
            shuffle_files=self._shuffle,
            read_config=read_config,
        )

        # CRITICAL: Force deterministic ordering on BOTH datasets.
        # dlimp's _apply_options() sets deterministic=False which causes file interleaving
        # to yield episodes in non-deterministic order, breaking the zip alignment.
        deterministic_options = tf.data.Options()
        deterministic_options.deterministic = True
        dataset = dataset.with_options(deterministic_options)
        latent_ds = latent_ds.with_options(deterministic_options)

        # Zip the two datasets (DLataset IS a tf.data.Dataset subclass)
        zipped = tf.data.Dataset.zip((dataset, latent_ds))

        def merge_episode(rlds_episode: dict, latent_episode: dict) -> dict:
            """Merge latent data into RLDS episode and verify alignment."""
            # Verify episode_index matches between RLDS and latent store.
            # RLDS stores episode_index per-step (broadcasted), so take first element.
            rlds_ep_idx = rlds_episode["episode_index"][0]
            latent_ep_idx = latent_episode["episode_index"]

            tf.debugging.assert_equal(
                rlds_ep_idx,
                latent_ep_idx,
                message="Episode index mismatch! RLDS and latent store are out of sync. "
                "This may indicate different shuffle seeds or shard ordering.",
            )

            # Store full latent sequences in trajectory (slicing happens later)
            # TFDS feature keys use underscore (e.g., "cam_0_latents"), convert to slash format
            for image_key in self._latent_manifest.image_keys:
                safe_key = image_key.replace("/", "_")
                rlds_episode[f"latents/{image_key}"] = tf.cast(latent_episode[f"{safe_key}_latents"], tf.bfloat16)

            # Broadcast scalar metadata to trajectory length (dlimp requires all fields to have
            # a leading trajectory dimension for flatten() to work)
            traj_len = rlds_episode["_len"][0]
            rlds_episode["_num_latent_frames"] = tf.repeat(latent_episode["num_latent_frames"], traj_len)
            rlds_episode["_latent_episode_index"] = tf.repeat(latent_episode["episode_index"], traj_len)
            return rlds_episode

        import dlimp as dl

        # Map and convert back to DLataset for continued pipeline processing
        merged = zipped.map(merge_episode, num_parallel_calls=self._num_parallel_calls)
        return dl.DLataset.from_tfds_dataset(merged, is_flattened=False)

    def _join_counterfactual_action_store(
        self,
        dataset,
        dataset_cfg: RLDSDataset,
        split: str,
        process_index: int,
        process_count: int,
    ):
        """Join counterfactual action store episodes with RLDS episodes at trajectory level.

        Uses identical TFDS loading settings as the RLDS dataset to ensure deterministic
        episode ordering for the zip.

        Args:
            dataset: The RLDS DLataset to join with.
            dataset_cfg: Dataset configuration (for validation).
            split: Split name (e.g., "train").
            process_index: This process's index for sharding.
            process_count: Total number of processes.

        Returns:
            Joined dataset where each trajectory has counterfactual actions attached.
        """
        import tensorflow as tf
        import tensorflow_datasets as tfds

        if self._ca_manifest is None:
            return dataset

        counterfactual_action_store.validate_manifest_against_rlds(
            self._ca_manifest,
            rlds_data_dir="",
            dataset_name=dataset_cfg.name,
            dataset_version=dataset_cfg.version,
        )

        # Validate strided action constraint once at init (not per-episode)
        if self._ca_manifest.stride > 1:
            max_offset = self._ca_manifest.stride - 1
            stored_action_horizon = self._ca_manifest.action_horizon
            required_horizon = max_offset + self._action_chunk_size
            if required_horizon > stored_action_horizon:
                raise ValueError(
                    f"Strided CA constraint violated: (stride-1) + target_action_horizon <= stored_action_horizon. "
                    f"Got: {max_offset} + {self._action_chunk_size} = {required_horizon} > {stored_action_horizon}"
                )

        ca_builder = counterfactual_action_store.get_counterfactual_action_store_builder(
            self._counterfactual_action_store_dir
        )

        split_to_use = (
            tfds.split_for_jax_process(split, process_index=process_index, process_count=process_count)
            if process_count > 1
            else split
        )

        read_config = tfds.ReadConfig(
            skip_prefetch=True,
            shuffle_seed=self._shuffle_seed if self._shuffle else None,
            num_parallel_calls_for_interleave_files=self._num_parallel_reads,
            interleave_cycle_length=self._num_parallel_reads,
        )

        ca_ds = ca_builder.as_dataset(
            split=split_to_use,
            shuffle_files=self._shuffle,
            read_config=read_config,
        )

        deterministic_options = tf.data.Options()
        deterministic_options.deterministic = True
        dataset = dataset.with_options(deterministic_options)
        ca_ds = ca_ds.with_options(deterministic_options)

        zipped = tf.data.Dataset.zip((dataset, ca_ds))

        def merge_episode(rlds_episode: dict, ca_episode: dict) -> dict:
            rlds_ep_idx = rlds_episode["episode_index"][0]
            ca_ep_idx = ca_episode["episode_index"]

            tf.debugging.assert_equal(
                rlds_ep_idx,
                ca_ep_idx,
                message="Episode index mismatch! RLDS and counterfactual action store are out of sync.",
            )

            counterfactual_actions = ca_episode["counterfactual_actions"]

            # Expand strided actions if stride > 1
            if self._ca_manifest.stride > 1:
                traj_len = rlds_episode["_len"][0]
                counterfactual_actions = counterfactual_action_store.expand_strided_counterfactual_actions(
                    counterfactual_actions,
                    stride=self._ca_manifest.stride,
                    num_steps=traj_len,
                    target_action_horizon=self._action_chunk_size,
                )

            # Store counterfactual actions in trajectory
            rlds_episode["counterfactual_actions"] = counterfactual_actions

            # Broadcast scalar metadata to trajectory length for dlimp flatten
            traj_len = rlds_episode["_len"][0]
            rlds_episode["_ca_episode_index"] = tf.repeat(ca_episode["episode_index"], traj_len)

            return rlds_episode

        import dlimp as dl

        merged = zipped.map(merge_episode, num_parallel_calls=self._num_parallel_calls)
        return dl.DLataset.from_tfds_dataset(merged, is_flattened=False)

    def _apply_latent_views(self, traj: dict) -> dict:
        """Slice latent sequences into per-step windows based on view configs.

        Called during trajectory processing after trajectory_transforms.

        Args:
            traj: Trajectory dict containing "latents/{image_key}" entries.

        Returns:
            Updated trajectory dict with sliced latent windows.
        """
        import tensorflow as tf

        if not self._latent_views or self._latent_manifest is None:
            return traj

        traj_len = tf.shape(traj["actions"])[0]
        latent_keys_to_remove = []

        for view_config in self._latent_views:
            view_image_keys = view_config.image_keys or tuple(self._short_to_full_key.keys())

            for short_key in view_image_keys:
                # Map short key to full manifest key for latent lookup
                full_key = self._short_to_full_key.get(short_key, short_key)
                latent_key = f"latents/{full_key}"
                if latent_key not in traj:
                    available_latent_keys = [k for k in traj if k.startswith("latents/")]
                    raise ValueError(
                        f"Latent key '{latent_key}' not found in trajectory. "
                        f"Available latent keys: {available_latent_keys}. "
                        f"Check that latent_store_dir contains latents for image key '{short_key}'."
                    )

                full_latents = traj[latent_key]  # [C, T_lat, H, W]

                # Build per-step sliced latents [T, C, window, H, W]
                per_step_latents = latent_store.slice_latents_for_trajectory(
                    full_latents,
                    traj_len,
                    self._latent_manifest.temporal_compression,
                    view_config.direction,
                    view_config.window_size,
                    view_config.stride,
                )

                # Use short key for output (e.g., "video_latents_cam_0")
                output_key = f"{view_config.output_key}_{short_key}"
                traj[output_key] = tf.cast(per_step_latents, tf.bfloat16)
                latent_keys_to_remove.append(latent_key)

        # Remove source latent sequences (they don't have trajectory length as leading dim,
        # which would cause dlimp's flatten() to fail)
        for key in latent_keys_to_remove:
            del traj[key]

        # Also remove scalar metadata that was only needed for latent processing
        traj.pop("_num_latent_frames", None)
        traj.pop("_latent_episode_index", None)

        return traj

    def trajectory_transforms(self, traj: dict, dataset_cfg: RLDSDataset) -> dict:
        """Per-trajectory processing. Override in subclasses.

        Default implementation extracts standard RLDS fields.
        Must produce a dict with at least an "actions" key for action chunking.
        """
        return {
            "actions": traj["action"],
            "observation": traj.get("observation", {}),
            "prompt": traj.get("language_instruction", ""),
        }

    def _prepare_trajectory(self, raw_traj: dict, dataset_cfg: RLDSDataset) -> dict:
        mapped_traj = self.trajectory_transforms(raw_traj, dataset_cfg)
        if "actions" not in mapped_traj:
            raise ValueError("trajectory_transforms must produce an 'actions' key for action chunking.")

        # Carry over latent data from raw trajectory if present
        for key, value in raw_traj.items():
            if key.startswith(("latents/", "_latent")):
                mapped_traj[key] = value

        # Carry over counterfactual action data if present
        if "counterfactual_actions" in raw_traj:
            mapped_traj["counterfactual_actions"] = raw_traj["counterfactual_actions"]
        for key in ("_ca_episode_index",):
            if key in raw_traj:
                mapped_traj[key] = raw_traj[key]

        # Apply latent view slicing before RL fields (so we can derive next_*)
        mapped_traj = self._apply_latent_views(mapped_traj)

        if self._critic_mode:
            return self._apply_rl_fields(raw_traj, mapped_traj, self._action_chunk_size)
        return mapped_traj

    def _apply_rl_fields(self, raw_traj: dict, mapped_traj: dict, action_chunk_size: int) -> dict:
        import tensorflow as tf

        per_step_rewards = self._get_per_step_rewards(raw_traj, mapped_traj)
        transformed_rewards = per_step_rewards * self._reward_scale + self._reward_bias
        per_step_terminations, per_step_truncations = self._get_per_step_done_flags(
            raw_traj, mapped_traj, per_step_rewards
        )
        per_step_dones = tf.logical_or(per_step_terminations, per_step_truncations)
        mc_returns = self._compute_mc_returns_tf(
            transformed_rewards, per_step_dones, self._discount, raw_traj=raw_traj, mapped_traj=mapped_traj
        )

        traj_len = tf.shape(mapped_traj["actions"])[0]
        chunk_rewards, chunk_terminations, chunk_truncations = self._compute_chunk_rl_fields(
            transformed_rewards, per_step_terminations, per_step_truncations, action_chunk_size, traj_len
        )

        next_indices = tf.minimum(tf.range(traj_len) + action_chunk_size, traj_len - 1)
        observation = mapped_traj.get("observation", {})
        if not isinstance(observation, dict):
            raise ValueError("trajectory_transforms must produce observation as a dict when critic_mode=True.")
        next_observation = {key: tf.gather(value, next_indices) for key, value in observation.items()}
        next_actions_raw = tf.gather(mapped_traj["actions"], next_indices)

        mapped_traj["reward"] = chunk_rewards
        mapped_traj["mc_return"] = mc_returns
        mapped_traj["termination"] = chunk_terminations
        mapped_traj["truncation"] = chunk_truncations
        mapped_traj["next_observation"] = next_observation
        mapped_traj["next_actions_raw"] = next_actions_raw
        if "counterfactual_actions" in mapped_traj:
            mapped_traj["counterfactual_next_actions"] = tf.gather(mapped_traj["counterfactual_actions"], next_indices)

        # Derive next_* for past-facing latent views
        if self._latent_views and self._latent_manifest is not None:
            for view_config in self._latent_views:
                if view_config.direction == "past":
                    image_keys = view_config.image_keys or self._latent_manifest.image_keys
                    for image_key in image_keys:
                        key = f"{view_config.output_key}_{image_key}"
                        if key in mapped_traj:
                            mapped_traj[f"next_{key}"] = tf.gather(mapped_traj[key], next_indices)

        return mapped_traj

    def _get_per_step_rewards(self, raw_traj: dict, mapped_traj: dict):
        del mapped_traj
        import tensorflow as tf

        if "reward" not in raw_traj:
            available_keys = ", ".join(sorted(raw_traj.keys()))
            raise ValueError(
                "Could not derive per-step rewards from trajectory: expected key 'reward'. "
                f"Available keys: {available_keys}"
            )
        return tf.cast(raw_traj["reward"], tf.float32)

    def _get_per_step_done_flags(self, raw_traj: dict, mapped_traj: dict, per_step_rewards):
        del mapped_traj
        del per_step_rewards
        import tensorflow as tf

        termination = self._find_first_bool_tensor(raw_traj, ("is_terminal", "terminal", "termination"))
        if termination is None:
            available_keys = ", ".join(sorted(raw_traj.keys()))
            raise ValueError(
                "Could not derive per-step termination flags from trajectory. Expected one of "
                "{is_terminal, terminal, termination}. "
                f"Available keys: {available_keys}"
            )
        truncation = self._find_first_bool_tensor(raw_traj, ("is_truncated", "truncated", "truncation"))
        if truncation is None:
            is_last = self._find_first_bool_tensor(raw_traj, ("is_last",))
            if is_last is None:
                available_keys = ", ".join(sorted(raw_traj.keys()))
                raise ValueError(
                    "Could not derive per-step truncation flags from trajectory. Expected one of "
                    "{is_truncated, truncated, truncation} or fallback key {is_last}. "
                    f"Available keys: {available_keys}"
                )
            truncation = tf.logical_and(is_last, tf.logical_not(termination))
        return termination, truncation

    def _find_first_bool_tensor(self, traj: dict, keys: Sequence[str]):
        import tensorflow as tf

        for key in keys:
            if key in traj:
                return tf.cast(traj[key], tf.bool)
        return None

    def _chunk_indices(self, traj_len, chunk_size):
        import tensorflow as tf

        base_indices = tf.broadcast_to(
            tf.range(chunk_size)[None],
            [traj_len, chunk_size],
        ) + tf.broadcast_to(
            tf.range(traj_len)[:, None],
            [traj_len, chunk_size],
        )
        valid_mask = base_indices < traj_len
        clamped_indices = tf.minimum(base_indices, traj_len - 1)
        return clamped_indices, valid_mask

    def _chunk_actions(self, traj: dict, action_chunk_size: int) -> dict:
        """Chunk action sequences and optional next_actions_raw."""
        import tensorflow as tf

        traj_len = tf.shape(traj["actions"])[0]
        action_chunk_indices, _ = self._chunk_indices(traj_len, action_chunk_size)
        traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
        if "next_actions_raw" in traj:
            traj["next_actions"] = tf.gather(traj["next_actions_raw"], action_chunk_indices)
            del traj["next_actions_raw"]
        return traj

    def _compute_chunk_rl_fields(self, rewards, terminations, truncations, chunk_size, traj_len):
        import tensorflow as tf

        chunk_indices, valid_mask = self._chunk_indices(traj_len, chunk_size)
        valid_mask_float = tf.cast(valid_mask, tf.float32)

        # Compute done flags and mask rewards after termination/truncation
        chunk_dones_raw = tf.gather(tf.logical_or(terminations, truncations), chunk_indices)
        chunk_dones_masked = tf.logical_and(chunk_dones_raw, valid_mask)
        chunk_dones_float = tf.cast(chunk_dones_masked, tf.float32)

        # Exclusive cumsum: cumsum[j] = sum of done[0:j-1]
        # If any done occurred before position j, this will be > 0
        cumsum_dones = tf.cumsum(chunk_dones_float, axis=1, exclusive=True)
        not_after_done_mask = tf.cast(tf.equal(cumsum_dones, 0.0), tf.float32)

        # Combine validity mask with termination mask
        reward_mask = valid_mask_float * not_after_done_mask

        chunk_rewards_raw = tf.gather(rewards, chunk_indices)
        discount_weights = tf.pow(self._discount, tf.cast(tf.range(chunk_size), tf.float32))
        chunk_rewards_masked = chunk_rewards_raw * reward_mask
        chunk_rewards = tf.reduce_sum(chunk_rewards_masked * discount_weights[None, :], axis=1)

        chunk_term_raw = tf.gather(terminations, chunk_indices)
        chunk_trunc_raw = tf.gather(truncations, chunk_indices)
        chunk_term_masked = tf.logical_and(chunk_term_raw, valid_mask)
        chunk_trunc_masked = tf.logical_and(chunk_trunc_raw, valid_mask)
        chunk_terminations = tf.reduce_any(chunk_term_masked, axis=1)
        chunk_truncations = tf.reduce_any(chunk_trunc_masked, axis=1)

        return chunk_rewards, chunk_terminations, chunk_truncations

    def _compute_mc_returns_tf(self, rewards, dones, discount, raw_traj=None, mapped_traj=None):
        del raw_traj, mapped_traj
        import tensorflow as tf

        rewards_rev = tf.reverse(rewards, axis=[0])
        dones_rev = tf.reverse(tf.cast(dones, tf.float32), axis=[0])

        def scan_fn(accumulated, inputs):
            reward, done = inputs
            return reward + discount * (1.0 - done) * accumulated

        mc_returns_rev = tf.scan(scan_fn, (rewards_rev, dones_rev), initializer=tf.constant(0.0, dtype=tf.float32))
        return tf.reverse(mc_returns_rev, axis=[0])

    def frame_transforms(self, frame: dict) -> dict:
        """Apply per-frame transforms. Override in subclasses for custom transforms.

        Default implementation decodes images from bytes to uint8 arrays.
        """
        import tensorflow as tf

        for key in self._image_obs_keys:
            if "observation" not in frame or key not in frame["observation"]:
                raise ValueError(
                    f"Configured image observation key '{key}' not found under frame['observation'] during decoding."
                )
            frame["observation"][key] = tf.io.decode_image(
                frame["observation"][key], expand_animations=False, dtype=tf.uint8
            )

        if "next_observation" in frame:
            for key in self._image_obs_keys:
                if key not in frame["next_observation"]:
                    raise ValueError(
                        f"Configured image observation key '{key}' not found under frame['next_observation'] during "
                        "decoding."
                    )
                frame["next_observation"][key] = tf.io.decode_image(
                    frame["next_observation"][key], expand_animations=False, dtype=tf.uint8
                )

        return frame

    def frame_filter(self, frame: dict) -> bool:
        """Filter predicate for frames. Override in subclasses to filter out invalid frames.

        Returns True to keep the frame, False to discard it.
        Default implementation keeps all frames.
        """
        import tensorflow as tf

        return tf.constant(value=True)

    def _apply_frame_transforms_to_trajectory(self, traj: dict) -> dict:
        """Apply frame_transforms to each timestep in a trajectory via tf.map_fn.

        Used in trajectory mode to ensure subclass frame_transforms overrides are applied.
        Returns a trajectory dict where each field is stacked across timesteps.
        """
        import tensorflow as tf

        traj_len = tf.shape(traj["actions"])[0]

        def extract_frame(t: int) -> dict:
            """Extract a single frame at timestep t."""
            frame: dict[str, Any] = {}
            for key, value in traj.items():
                if isinstance(value, dict):
                    frame[key] = {k: v[t] for k, v in value.items()}
                else:
                    frame[key] = value[t]
            return frame

        def process_frame(t: int) -> dict:
            """Extract frame at t and apply frame_transforms."""
            frame = extract_frame(t)
            return self.frame_transforms(frame)

        # Use tf.map_fn to apply frame_transforms to each timestep
        # We need to get the output structure from a sample frame
        sample_frame = process_frame(0)

        def map_body(t):
            return process_frame(t)

        # Map over all timesteps
        return tf.map_fn(
            map_body,
            tf.range(traj_len),
            fn_output_signature=tf.nest.map_structure(
                lambda x: tf.TensorSpec(shape=x.shape, dtype=x.dtype), sample_frame
            ),
        )

    def __iter__(self):
        """Iterate over the dataset.

        In training mode: yields batched frame dicts.
        In trajectory mode: yields trajectory dicts (stacked arrays per trajectory).
        """
        yield from self.dataset.as_numpy_iterator()

    def __len__(self) -> int:
        """Return dataset length.

        In trajectory mode: returns number of trajectories (striding already applied at TFDS level).
        In training mode: raises TypeError (infinite streaming has no length).
        """
        if self._return_trajectories:
            return int(self.dataset.cardinality().numpy())
        raise TypeError("Training mode dataset has no length (infinite streaming).")
