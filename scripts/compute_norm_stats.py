"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import pathlib

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.rlds_dataset as rlds_dataset
import openpi.transforms as transforms

# Chunk length used for action_diff per-step stats. Sized to cover the longest
# action_horizon any downstream consumer might use (currently 50 for the
# robocoin_bimanual_paligemma_q_sarsa value function); shorter consumers slice
# the resulting (50, D) array down to their own horizon at config-create time.
ACTION_DIFF_HORIZON = 50


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_numpy_dataloader(
    data_config: _config.DataConfig,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    """Create dataloader for NumpyDataset from Minari or legacy D4RL."""
    if data_config.minari_dataset_id is not None:
        dataset = _data_loader.create_numpy_dataset_from_minari(
            data_config.minari_dataset_id,
            discount=data_config.discount,
        )
    elif data_config.legacy_d4rl_env_name is not None:
        dataset = _data_loader.create_numpy_dataset_from_legacy_d4rl(
            data_config.legacy_d4rl_env_name,
            discount=data_config.discount,
            reward_scale=data_config.reward_scale,
            reward_bias=data_config.reward_bias,
        )
    else:
        raise ValueError("Data config must have minari_dataset_id or legacy_d4rl_env_name")

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=0,
    )
    return data_loader, num_batches


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)

    # Filter out RL-specific transforms - we only need raw state/actions for norm stats
    input_transforms = []
    for t in data_config.data_transforms.inputs:
        # Skip transforms that require RL field computation
        transform_name = type(t).__name__
        if transform_name == "ValueFunctionInputs":
            continue
        input_transforms.append(t)

    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *input_transforms,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None:
        # Training-mode RLDS streams infinitely → len(dataset) raises. Use max_frames
        # directly to compute batch count.
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    fine_tune: str | None = None,
    max_frames: int | None = 200_000,
    dataset_name: str | None = None,
    output_dir: str | None = None,
):
    """Compute normalization statistics for a config.

    Args:
        config_name: name of a registered TrainConfig.
        fine_tune: optional name of a registered FineTuneConfig. When provided, its
            data/model overrides are applied to the base TrainConfig before computing
            stats — so the stats reflect the dataset the fine-tune actually trains on
            (e.g. RoboCasa data when fine-tuning a RoboCOIN base).
        max_frames: cap on number of frames to iterate. Defaults to 200_000 — enough
            for stable mean/std/quantile estimates on RoboCasa-scale data without
            relying on RLDS dataset length (the script's RLDS branch hard-codes
            length for DROID and is otherwise unreliable).
        dataset_name: optional "name:version" or "name" string. When set, replaces
            the data factory's `datasets` field with a single-element tuple referring
            to this RLDS dataset. Useful for computing per-task norm stats without
            registering a new TrainConfig per task.
        output_dir: optional explicit directory to write `norm_stats.json` to.
            Overrides the default `config.assets_dirs / repo_id` path.
    """
    config = _config.get_config(config_name)
    if fine_tune is not None:
        ft_config = _config.get_fine_tune_config(fine_tune)
        config = ft_config.apply_overrides(config)
        print(f"Applied FineTuneConfig overrides from '{fine_tune}'")

    if dataset_name is not None:
        if ":" in dataset_name:
            ds_name, ds_version = dataset_name.split(":", 1)
        else:
            ds_name, ds_version = dataset_name, "1.0.0"
        new_dataset = rlds_dataset.RLDSDataset(name = ds_name, version = ds_version, weight = 1.0)
        config = dataclasses.replace(
            config, data = dataclasses.replace(config.data, datasets = (new_dataset,))
        )
        print(f"Overrode dataset to {ds_name}:{ds_version}")

    # The script computes two flavors of action stats: absolute action[0] (1-D) and
    # the chunk-wise-delta of the whole chunk (2-D, per-timestep). Force the data
    # pipeline to emit absolute actions so the DeltaActions transform applied here
    # is the only source of deltas — even if the config has use_chunk_wise_delta=True
    # for training.
    if getattr(config.data, "use_chunk_wise_delta", False):
        config = dataclasses.replace(
            config, data = dataclasses.replace(config.data, use_chunk_wise_delta = False),
        )
        print("Overrode use_chunk_wise_delta=False for stats computation.")

    data_config = config.data.create(config.assets_dirs, config.model)

    # Always pull 50-step chunks for action_diff stats so the produced (H=50, D) array
    # covers the longest action_horizon any consumer might use. Runtime configs slice
    # this down to their own action_horizon at data_config.create() time.
    action_horizon = ACTION_DIFF_HORIZON

    # Determine what type of data loader to use
    if data_config.minari_dataset_id is not None:
        data_loader, num_batches = create_numpy_dataloader(data_config, config.batch_size, max_frames)
        output_id = data_config.minari_dataset_id.replace("/", "_")
    elif data_config.legacy_d4rl_env_name is not None:
        data_loader, num_batches = create_numpy_dataloader(data_config, config.batch_size, max_frames)
        output_id = data_config.asset_id
    elif data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, action_horizon, config.batch_size, max_frames
        )
        output_id = data_config.repo_id
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )
        output_id = data_config.repo_id

    # DeltaActions for the action_diff stats. Hardcoded for the RoboCasa policy
    # arm-first bimanual layout (14-D = [arm_pos(3), arm_rpy(3), arm_grip(1), zeros(7)]).
    # Position dims get linear delta, rpy block at index 3 gets relative-rotation delta,
    # the gripper and the trailing zero placeholder slots stay absolute.
    action_diff_transform = transforms.DeltaActions(
        mask = transforms.make_bool_mask(6, -1, -7),
        rpy_index_start = (3,),
    )

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()  # populated only with action[0] -> (D,)
    action_diff_stats: list[normalize.RunningStats] = []  # one RunningStats per timestep -> (H, D)

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        state = np.asarray(batch["state"])      # (B, D_state)
        actions = np.asarray(batch["actions"])  # (B, H, D_act)
        state_stats.update(state)
        # Use only the first action of the chunk for the absolute-action stats.
        action_stats.update(actions[:, 0, :])

        # Apply DeltaActions to the absolute action chunk to get per-step deltas.
        delta = action_diff_transform({"state": state, "actions": actions})["actions"]  # (B, H, D_act)
        if not action_diff_stats:
            action_diff_stats = [normalize.RunningStats() for _ in range(delta.shape[1])]
        for t in range(delta.shape[1]):
            action_diff_stats[t].update(delta[:, t, :])

    # Stack per-timestep stats into (H, D) arrays so action_diff is 2-D, not collapsed.
    diff_per_t = [s.get_statistics() for s in action_diff_stats]
    action_diff_norm = normalize.NormStats(
        mean = np.stack([s.mean for s in diff_per_t], axis = 0),
        std = np.stack([s.std for s in diff_per_t], axis = 0),
        q01 = np.stack([s.q01 for s in diff_per_t], axis = 0),
        q99 = np.stack([s.q99 for s in diff_per_t], axis = 0),
    )

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
        "action_diff": action_diff_norm,
    }

    if output_dir is not None:
        output_path = pathlib.Path(output_dir)
    else:
        output_path = config.assets_dirs / output_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
