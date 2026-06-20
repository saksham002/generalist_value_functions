"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import numpy as np
import pathlib
from typing import Any, ClassVar, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.best_of_n as _best_of_n
import openpi.models.mlp_config as mlp_config
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tanh_gaussian as _tanh_gaussian
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.d4rl_policy as d4rl_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.robocasa_policy as robocasa_policy
import openpi.robocoin_utils.utils as _robocoin_utils
from openpi.policy_extraction import objectives as _policy_extraction
import openpi.shared.download as _download

try:
    import openpi.shared.legacy_d4rl_utils as legacy_d4rl_utils
except Exception:
    legacy_d4rl_utils = None  # type: ignore
import openpi.shared.minari_utils as minari_utils
from openpi.shared.action_bounds import ActionBounds
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.hdf5_rlds_dataset as hdf5_rlds_dataset
import openpi.training.lerobot_rlds_dataset as lerobot_rlds_dataset
import openpi.training.robocoin_rlds_dataset as robocoin_rlds_dataset
import openpi.training.rlds_dataset as rlds_dataset
import openpi.training.robocasa_datasets as robocasa_datasets
import openpi.training.robocasa_rlds_dataset as robocasa_rlds_dataset
import openpi.training.state_action_spaces as state_action_spaces
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import openpi.value_functions.base_value_functions as _value_functions_base
import openpi.value_functions.heads as _heads
import openpi.value_functions.networks.ensemble as _ensemble_network
import openpi.value_functions.networks.mlp as _mlp_network
import openpi.value_functions.networks.paligemma as _paligemma_network
import openpi.value_functions.value_function as _value_function
import openpi.value_functions.value_transforms as _value_transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Optional override RLDS dir for validation trajectory caching only. When set,
    # `val_trajectory_dataset` reads from here instead of `rlds_data_dir` so we
    # can use a lighter (e.g. resized) variant for val without affecting training.
    val_dataset_dir: str | None = None
    rlds_dataset_class: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[rlds_dataset.RLDSDataset] = ()
    robocoin_use_eef: bool = False
    val_split: str = "test"
    clip_normalized_bounds: dict[str, tuple[float, float]] | None = None
    counterfactual_action_store_dir: str | None = None
    max_num_demos: int | None = None
    rlds_kwargs: dict[str, Any] = dataclasses.field(default_factory = dict)

    # RL training mode options (for value function training)
    critic_mode: bool = False  # If True, use value function training pipeline
    discount: float = 0.99  # Discount factor (used if MC returns not in dataset)

    # Reward transformation: r' = reward_scale * r + reward_bias (applied before MC return computation)
    reward_scale: float = 1.0
    reward_bias: float = 0.0

    # If set, load data directly from Minari dataset (fastest option for in-memory datasets)
    minari_dataset_id: str | None = None

    # If set, load data directly from legacy D4RL dataset (alternative to Minari)
    legacy_d4rl_env_name: str | None = None

    # Multi-transition training options
    # Number of transitions per sample (for multi-state value functions)
    num_transitions_per_sample: int | None = None
    # Sampler type: "uniform", "trajectory_uniform", "trajectory_ordered", or "trajectory_consecutive"
    multi_transition_sampler_type: Literal[
        "uniform", "trajectory_uniform", "trajectory_ordered", "trajectory_consecutive"
    ] = "trajectory_uniform"

    # Upsampling weight for transitions with reward=1. If > 1.0, these transitions
    # will be sampled more frequently. A value of 2.0 means reward=1 transitions
    # are sampled twice as often as other transitions.
    reward_1_upsample_weight: float = 1.0

    # Keys to skip during normalization/unnormalization
    skip_normalize_keys: tuple[str, ...] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(
                            model_config.action_dim,
                            action_dim_offset = model_config.action_dim_offset,
                            pad_state = model_config.pad_state_to_action_dim,
                            action_dim_mask = model_config.action_dim_mask,
                        ),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )
            case _model.ModelType.MLP:
                # MLP model: no tokenization or image processing.
                # Dimensions are handled by the model config and dataset loader.
                return _transforms.Group(inputs=[])


@dataclasses.dataclass(frozen=True)
class DecodeRoboCoinPromptBytes:
    """Decode RoboCOIN prompt bytes before generic tokenization."""

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        for key in ("prompt", "subtask_text"):
            if key not in data:
                continue
            value = data[key]
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            data[key] = value
        return data


@dataclasses.dataclass(frozen=True)
class AddValidationVariants:
    """Add validation variants (random actions + optional negative-text prompt)."""

    tokenizer: _tokenizer.PaligemmaTokenizer | _tokenizer.Gemma3Tokenizer | _tokenizer.Gemma4Tokenizer
    use_quantile_norm: bool = False
    # When True (default), produce a negative-prompt variant via the RoboCOIN-specific
    # `generate_negative_subtask_text` heuristic; requires "subtask_1" to be present.
    # Set False for non-RoboCOIN datasets that only need random_actions.
    include_negative: bool = True

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "actions" not in data:
            return data

        traj_index = None
        frame_index = None
        if "_traj_index" in data:
            traj_index = int(np.asarray(data["_traj_index"]).item())
        if "_frame_index" in data:
            frame_index = int(np.asarray(data["_frame_index"]).item())

        if self.include_negative and "subtask_1" in data:
            subtask_text = _robocoin_utils.decode_text(data["subtask_1"])
            negative_subtask_text = _robocoin_utils.generate_negative_subtask_text(subtask_text)
            negative_tokens, negative_mask = self.tokenizer.tokenize(negative_subtask_text, state = None)
            data["negative_subtask_1_text"] = negative_subtask_text
            data["tokenized_negative_prompt"] = negative_tokens
            data["tokenized_negative_prompt_mask"] = negative_mask

        data["random_actions"] = _robocoin_utils.sample_random_actions(
            np.asarray(data["actions"]),
            use_quantile_norm = self.use_quantile_norm,
            _traj_index = traj_index,
            frame_index = frame_index,
        )
        return data


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            # Always re-download norm_stats: the files are tiny and stale per-pod caches
            # silently break runs when the on-disk schema changes (e.g. action_diff added).
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir, force_download = True))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class D4RLDataConfig(DataConfigFactory):
    """
    Data config for D4RL datasets (state-only, no images).

    D4RL datasets contain state observations and actions only.
    To convert D4RL data to LeRobot format, see examples/d4rl/convert_d4rl_to_lerobot.py

    When critic_mode=True, the data loader will:
    1. Load transitions as SARSA tuples: (s, a, r, s', a')
    2. Compute discounted Monte-Carlo returns for each trajectory
    3. Include 'mc_return' in the data dict for value function training
    """

    # Action dimension for the D4RL environment. If None, will be inferred from model_config.
    action_dim: int | None = None
    # Default task name (environment name) if not provided in data
    default_task: str | None = None

    # RL training mode options
    critic_mode: bool = False  # If True, load SARSA tuples + MC returns
    discount: float = 0.99  # Discount factor for MC return computation

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # No repack needed - D4RL LeRobot datasets already have 'state' and 'actions' keys
        repack_transform = _transforms.Group(inputs=[])

        if self.critic_mode:
            # Critic mode: use value function transforms
            # All RL fields are stored in the dataset (computed at conversion time)
            data_transforms = _transforms.Group(
                inputs=[_value_transforms.ValueFunctionInputs()],
                outputs=[],
            )
            # Value function configs don't have model_type, skip model transforms
            model_transforms = _transforms.Group(inputs=[], outputs=[])
        else:
            # Standard policy training mode - action_dim required
            action_dim = self.action_dim if self.action_dim is not None else model_config.action_dim
            data_transforms = _transforms.Group(
                inputs=[d4rl_policy.D4RLInputs()],
                outputs=[d4rl_policy.D4RLOutputs(action_dim=action_dim)],
            )
            model_transforms = ModelTransformFactory(default_prompt=self.default_task)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            critic_mode=self.critic_mode,
            discount=self.discount,
        )


@dataclasses.dataclass(frozen=True)
class MinariDataConfig(DataConfigFactory):
    """Data config for direct Minari dataset loading (fast, in-memory).

    This config enables the numpy data loader path, which loads the entire
    dataset into memory for maximum training speed. Best for state-only
    datasets that fit in RAM (e.g., D4RL/Minari antmaze, locomotion).

    MC returns and RL fields are computed on-the-fly when loading the dataset.
    """

    # Minari dataset ID (e.g., 'D4RL/antmaze/large-diverse-v1')
    minari_dataset_id: str = tyro.MISSING
    # Discount factor for MC return computation
    discount: float = 0.99
    # Reward transformation: r' = reward_scale * r + reward_bias
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # Upsampling weight for transitions with reward=1 (see DataConfig.reward_1_upsample_weight)
    reward_1_upsample_weight: float = 1.0
    # Keys to skip during normalization (default: skip all to disable normalization)
    skip_normalize_keys: tuple[str, ...] = (
        "state",
        "actions",
        "next_state",
        "next_actions",
    )

    # Override repo_id from parent - not used for minari loading
    repo_id: str = "minari"  # Dummy value, not used

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Value function transforms for RL training
        data_transforms = _transforms.Group(
            inputs=[_value_transforms.ValueFunctionInputs()],
            outputs=[],
        )
        model_transforms = _transforms.Group(inputs=[], outputs=[])

        asset_id = self.minari_dataset_id.replace("/", "_")
        norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)

        # Add next_state and next_actions with same normalization as their current counterparts
        if norm_stats is not None:
            if "state" in norm_stats:
                norm_stats["next_state"] = norm_stats["state"]
            if "actions" in norm_stats:
                norm_stats["next_actions"] = norm_stats["actions"]
                norm_stats["counterfactual_actions"] = norm_stats["actions"]
                norm_stats["counterfactual_next_actions"] = norm_stats["actions"]

            # Filter out keys that should be skipped during normalization
            if self.skip_normalize_keys:
                norm_stats = {k: v for k, v in norm_stats.items() if k not in self.skip_normalize_keys}

        return DataConfig(
            repo_id=None,  # Not using LeRobot
            asset_id=asset_id,
            norm_stats=norm_stats,
            repack_transforms=_transforms.Group(inputs=[]),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=False,
            critic_mode=True,
            discount=self.discount,
            reward_scale=self.reward_scale,
            reward_bias=self.reward_bias,
            minari_dataset_id=self.minari_dataset_id,
            reward_1_upsample_weight=self.reward_1_upsample_weight,
            skip_normalize_keys=self.skip_normalize_keys,
        )


@dataclasses.dataclass(frozen=True)
class MultiTransitionMinariDataConfig(MinariDataConfig):
    """Data config for multi-transition value function training.

    Extends MinariDataConfig to sample multiple transitions per sample,
    enabling training of multi-state value functions.
    """

    # Number of transitions per sample
    num_transitions_per_sample: int = tyro.MISSING
    # Sampler type: uniform, trajectory_uniform, trajectory_ordered, or trajectory_consecutive
    multi_transition_sampler_type: Literal[
        "uniform", "trajectory_uniform", "trajectory_ordered", "trajectory_consecutive"
    ] = "trajectory_uniform"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Get base config from parent
        base_config = super().create(assets_dirs, model_config)

        # Add multi-transition settings
        return dataclasses.replace(
            base_config,
            num_transitions_per_sample=self.num_transitions_per_sample,
            multi_transition_sampler_type=self.multi_transition_sampler_type,
        )


@dataclasses.dataclass(frozen=True)
class LegacyD4RLDataConfig(DataConfigFactory):
    """Data config for direct legacy D4RL dataset loading (fast, in-memory).

    This config enables loading datasets from the d4rl library directly,
    providing an alternative to Minari for users who prefer the original
    D4RL interface or need access to older dataset versions.

    Supports special handling for sparse reward environments (antmaze)
    where failed trajectories use reward_neg / (1-gamma) as MC returns.
    """

    # D4RL environment name (e.g., 'antmaze-large-diverse-v2')
    legacy_d4rl_env_name: str = tyro.MISSING
    # Discount factor for MC return computation
    discount: float = 0.99
    # Reward transformation: r' = reward_scale * r + reward_bias
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # Action clipping margin (clips to [-clip_action, clip_action])
    clip_action: float = 0.999
    # Upsampling weight for transitions with reward=1
    reward_1_upsample_weight: float = 1.0
    # Keys to skip during normalization (default: skip all to disable normalization)
    skip_normalize_keys: tuple[str, ...] = (
        "state",
        "actions",
        "next_state",
        "next_actions",
    )

    # Override repo_id from parent - not used for d4rl loading
    repo_id: str = "legacy_d4rl"  # Dummy value, not used

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Value function transforms for RL training
        data_transforms = _transforms.Group(
            inputs=[_value_transforms.ValueFunctionInputs()],
            outputs=[],
        )
        model_transforms = _transforms.Group(inputs=[], outputs=[])

        # Use env name as asset_id (replace hyphens with underscores)
        asset_id = self.legacy_d4rl_env_name.replace("-", "_")
        norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)

        # Add next_state and next_actions with same normalization as their current counterparts
        if norm_stats is not None:
            if "state" in norm_stats:
                norm_stats["next_state"] = norm_stats["state"]
            if "actions" in norm_stats:
                norm_stats["next_actions"] = norm_stats["actions"]
                norm_stats["counterfactual_actions"] = norm_stats["actions"]
                norm_stats["counterfactual_next_actions"] = norm_stats["actions"]

            # Filter out keys that should be skipped during normalization
            if self.skip_normalize_keys:
                norm_stats = {k: v for k, v in norm_stats.items() if k not in self.skip_normalize_keys}

        return DataConfig(
            repo_id=None,  # Not using LeRobot
            asset_id=asset_id,
            norm_stats=norm_stats,
            repack_transforms=_transforms.Group(inputs=[]),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=False,
            critic_mode=True,
            discount=self.discount,
            reward_scale=self.reward_scale,
            reward_bias=self.reward_bias,
            legacy_d4rl_env_name=self.legacy_d4rl_env_name,
            reward_1_upsample_weight=self.reward_1_upsample_weight,
            skip_normalize_keys=self.skip_normalize_keys,
        )


@dataclasses.dataclass(frozen=True)
class MultiTransitionLegacyD4RLDataConfig(LegacyD4RLDataConfig):
    """Data config for multi-transition value function training with legacy D4RL.

    Extends LegacyD4RLDataConfig to sample multiple transitions per sample,
    enabling training of multi-state value functions.
    """

    # Number of transitions per sample
    num_transitions_per_sample: int = tyro.MISSING
    # Sampler type: uniform, trajectory_uniform, trajectory_ordered, or trajectory_consecutive
    multi_transition_sampler_type: Literal[
        "uniform", "trajectory_uniform", "trajectory_ordered", "trajectory_consecutive"
    ] = "trajectory_uniform"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Get base config from parent
        base_config = super().create(assets_dirs, model_config)

        # Add multi-transition settings
        return dataclasses.replace(
            base_config,
            num_transitions_per_sample=self.num_transitions_per_sample,
            multi_transition_sampler_type=self.multi_transition_sampler_type,
        )


@dataclasses.dataclass(frozen=True)
class RoboCoinRldsDataConfig(DataConfigFactory):
    """Data config for RoboCOIN using the RLDS dataset pipeline."""

    repo_id: str = "robocoin"
    assets: AssetsConfig = dataclasses.field(
        default_factory = lambda: AssetsConfig(
            assets_dir = "gs://saksham-euw4/robocoin_bimanual/norm_stats",
            asset_id = "embodiment_wise",
        )
    )

    # RLDS dataset loading
    rlds_data_dir: str = "gs://saksham-euw4/robocoin_bimanual"
    # Optional override RLDS dir used only for validation trajectory caching.
    # Useful when training reads from an _unresized variant whose 480x720 raw
    # images blow up host RAM during val caching; set this to a resized variant.
    val_dataset_dir: str | None = None
    datasets: Sequence[rlds_dataset.RLDSDataset] = (
        rlds_dataset.RLDSDataset(name = "robocoin_bimanual", version = "1.0.0", weight = 1.0),
    )
    val_split: str = "val"
    counterfactual_action_store_dir: str | None = None
    max_num_demos: int | None = None
    shuffle_buffer_size: int = 250_000
    num_parallel_reads: int = 8
    num_parallel_calls: int = 8

    # Image and model
    image_size: tuple[int, int] = (224, 224)
    max_token_len: int = 48

    # RL training
    discount: float = 0.99
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    critic_mode: bool = True

    # Data pipeline options
    td_n: int | None = None
    use_eef: bool = False
    use_quantile_norm: bool = False
    filter_n: int | None = None
    mask_50fps: bool = False
    mask_boundary_actions: bool = True
    replace_boundary_actions: bool = False
    variable_horizon: bool = False
    # Lower bound (50-fps frames, fps-scaled like the upper cap) on the sampled chunk length when variable_horizon=True.
    lower_action_horizon: int = 1
    use_chunk_wise_delta: bool = False
    state_dim: int = 14
    subtask_prompt_mode: robocoin_rlds_dataset.SubtaskPromptMode = "subtask_only"

    def __post_init__(self) -> None:
        if self.mask_boundary_actions and self.replace_boundary_actions:
            raise ValueError("At most one of mask_boundary_actions and replace_boundary_actions can be True.")

    @override
    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict | None:
        """Load norm stats, handling both standard openpi format and RoboCOIN-specific format.

        Standard format: {"norm_stats": {"state": {"mean": [...], ...}, ...}}
        RoboCOIN format: {"embodiment": {"observation.state": {"mean": [...], ...}, "action": {...}, ...}, ...}

        For RoboCOIN format, converts key names (observation.state → state, action → actions) and applies
        EEF combining / chunk-wise delta selection based on config flags.
        """
        if asset_id is None:
            return None

        import json

        path = epath.Path(str(assets_dir / asset_id)) / "norm_stats.json"
        if not path.exists():
            logging.info(f"Norm stats not found at {path}, skipping.")
            return None

        data = json.loads(path.read_text())

        if "norm_stats" in data:
            return _normalize.deserialize_json(path.read_text())

        # Flat single-embodiment RoboCOIN format: "observation.state" is a top-level key
        if "observation.state" in data:
            result = self._convert_robocoin_stats(data)
            logging.info(f"Loaded flat single-embodiment RoboCOIN norm_stats from {path}")
            return result

        # RoboCOIN format: top-level keys are embodiment names
        result = {}
        for embodiment, emb_data in data.items():
            if not isinstance(emb_data, dict) or "observation.state" not in emb_data:
                continue
            result[embodiment] = self._convert_robocoin_stats(emb_data)

        if not result:
            return None

        logging.info(f"Loaded embodiment-keyed RoboCOIN norm_stats from {path}, embodiments: {list(result.keys())}")
        return result

    def _convert_robocoin_stats(self, data: dict) -> dict[str, _transforms.NormStats]:
        """Convert a single embodiment's RoboCOIN-format stats to NormStats dict."""
        import numpy as np

        norm_stats: dict[str, _transforms.NormStats] = {}

        state_stats = data["observation.state"]
        raw_state_stats = _transforms.NormStats(
            mean = np.array(state_stats["mean"]),
            std = np.array(state_stats["std"]),
            q01 = np.array(state_stats["q01"]) if self.use_quantile_norm else None,
            q99 = np.array(state_stats["q99"]) if self.use_quantile_norm else None,
        )
        state_norm_stats = raw_state_stats
        if self.use_eef:
            eef_state_stats = data["eef_sim_pose_state"]
            combined_state = {}
            stat_keys = ("mean", "std", "q01", "q99") if self.use_quantile_norm else ("mean", "std")
            for stat_key in stat_keys:
                eef_arr = np.array(eef_state_stats[stat_key])
                joint_arr = np.array(state_stats[stat_key])
                combined_state[stat_key] = self._combine_eef_and_gripper_stats(eef_arr, joint_arr)
            state_norm_stats = _transforms.NormStats(**combined_state)
        if self.use_eef:
            assert state_norm_stats.mean.shape[-1] == 14, (
                f"use_eef=True requires 14D state norm stats inside the data pipeline, "
                f"got {state_norm_stats.mean.shape[-1]}D"
            )
            norm_stats["state"] = state_norm_stats
        elif self.state_dim == 14:
            assert state_norm_stats.mean.shape[-1] == 14, (
                f"state_dim=14 but norm stats have {state_norm_stats.mean.shape[-1]}D state"
            )
            norm_stats["state"] = state_norm_stats
        elif self.state_dim == 16:
            if state_norm_stats.mean.shape[-1] == 16:
                norm_stats["state"] = state_norm_stats
            else:
                norm_stats["state"] = self._pad_state_norm_stats_14_to_16(state_norm_stats)
        else:
            raise ValueError(f"Unsupported state_dim={self.state_dim}, expected 14 or 16")

        eef_action_key = "eef_sim_pose_action_diff" if self.use_chunk_wise_delta else "eef_sim_pose_action"
        ax = -1 if self.use_chunk_wise_delta else 0

        if "action" in data:
            if self.use_eef or self.use_chunk_wise_delta:
                # Chunk-wise delta actions are 14D EEF-layout (6 pose + gripper + 6 pose + gripper);
                # non-gripper dims come from eef_sim_pose_action_diff and gripper dims (6, 13) stay
                # absolute from "action". `*_diff` stats can carry an extra leading chunk dim, so
                # broadcast the absolute gripper slice to match before concat.
                eef_action_stats = data[eef_action_key]
                gripper_action_stats = data["action"]
                combined = {}
                stat_keys = ("mean", "std", "q01", "q99") if self.use_quantile_norm else ("mean", "std")
                for stat_key in stat_keys:
                    eef_arr = np.array(eef_action_stats[stat_key])
                    abs_arr = np.array(gripper_action_stats[stat_key])
                    # Grippers sit at dim//2 - 1 (left) and dim - 1 (right) of the raw action,
                    # matching `_construct_eef_repr`. Works for both 14D (6, 13) and 16D (7, 15).
                    raw_action_dim = abs_arr.shape[-1]
                    left_gripper_idx = raw_action_dim // 2 - 1
                    right_gripper_idx = raw_action_dim - 1
                    left_grip = self._broadcast_gripper_stats(
                        abs_arr[..., left_gripper_idx:left_gripper_idx + 1], eef_arr[..., :1]
                    )
                    right_grip = self._broadcast_gripper_stats(
                        abs_arr[..., right_gripper_idx:right_gripper_idx + 1], eef_arr[..., :1]
                    )
                    combined[stat_key] = np.concatenate(
                        [
                            eef_arr[..., :6],
                            left_grip,
                            eef_arr[..., 6:12],
                            right_grip,
                        ],
                        axis = ax,
                    )
                norm_stats["actions"] = _transforms.NormStats(**combined)
            else:
                action_stats = data["action"]
                norm_stats["actions"] = _transforms.NormStats(
                    mean = np.array(action_stats["mean"]),
                    std = np.array(action_stats["std"]),
                    q01 = np.array(action_stats["q01"]) if self.use_quantile_norm else None,
                    q99 = np.array(action_stats["q99"]) if self.use_quantile_norm else None,
                )

        if "state" in norm_stats:
            norm_stats["next_state"] = norm_stats["state"]
        if "actions" in norm_stats:
            norm_stats["next_actions"] = norm_stats["actions"]
            # Cached counterfactual actions stay absolute+unnormalized on disk; DeltaActions now
            # converts them to chunk-wise-delta and these aliases quantile-normalize them with the
            # same per-embodiment action stats as next_actions (Normalize selects by embodiment).
            norm_stats["counterfactual_actions"] = norm_stats["actions"]
            norm_stats["counterfactual_next_actions"] = norm_stats["actions"]

        return norm_stats

    @staticmethod
    def _broadcast_gripper_stats(abs_slice, ref_slice):
        """Broadcast absolute gripper stats to the rank of reference (possibly chunked) stats."""
        import numpy as np

        if abs_slice.ndim < ref_slice.ndim:
            return np.broadcast_to(abs_slice, ref_slice.shape).copy()
        return abs_slice

    @staticmethod
    def _combine_eef_and_gripper_stats(eef_arr, joint_arr):
        """Combine EEF pose stats with the left/right gripper slots from the joint-state stats.

        EEF stats are always 12D (xyz+rpy per arm); the grippers come from the joint-state
        stats at dim//2-1 (left) and dim-1 (right), matching the layout produced by the
        RLDS dataset's `_construct_eef_repr` helper.
        """
        import numpy as np

        total_dim = joint_arr.shape[-1]
        assert eef_arr.shape[-1] == 12, (
            f"Expected EEF stats to have 12 dims (xyz+rpy per arm), got {eef_arr.shape}"
        )
        left_gripper_index = total_dim // 2 - 1
        right_gripper_index = total_dim - 1

        return np.concatenate(
            [
                eef_arr[..., :6],
                joint_arr[..., left_gripper_index:left_gripper_index + 1],
                eef_arr[..., 6:12],
                joint_arr[..., right_gripper_index:right_gripper_index + 1],
            ],
            axis = -1,
        )

    @staticmethod
    def _pad_14_to_16(x, fill_value):
        """Pad 14D state vector to 16D: x[:7], fill, x[7:], fill."""
        import numpy as np

        return np.concatenate([x[..., :6], np.full((*x.shape[:-1], 1), fill_value), x[..., 6:13], np.full((*x.shape[:-1], 1), fill_value), x[..., 13:]], axis = -1)

    def _pad_state_norm_stats_14_to_16(self, stats: _transforms.NormStats) -> _transforms.NormStats:
        """Pad 14D norm stats to 16D with identity-like fill values."""
        return _transforms.NormStats(
            mean = self._pad_14_to_16(stats.mean, 0.0),
            std = self._pad_14_to_16(stats.std, 1.0),
            q01 = self._pad_14_to_16(stats.q01, -1.0),
            q99 = self._pad_14_to_16(stats.q99, 1.0),
        )

    def _create_clip_normalized_bounds(self) -> dict[str, tuple[float, float]]:
        clip_bound = 1.25 if self.use_quantile_norm else 5.0
        return {
            "state": (-clip_bound, clip_bound),
            "actions": (-clip_bound, clip_bound),
            "next_state": (-clip_bound, clip_bound),
            "next_actions": (-clip_bound, clip_bound),
            "counterfactual_actions": (-clip_bound, clip_bound),
            "counterfactual_next_actions": (-clip_bound, clip_bound),
        }

    def _get_critic_tokenizer(
        self, model_config: _model.BaseModelConfig
    ) -> _tokenizer.PaligemmaTokenizer | _tokenizer.Gemma3Tokenizer | _tokenizer.Gemma4Tokenizer | None:
        network_config = self._get_critic_network_config(model_config)
        if isinstance(network_config, _paligemma_network.PaliGemmaNetworkConfig):
            return network_config.get_tokenizer(max_len = self.max_token_len)
        return None

    def _get_critic_network_config(self, model_config: _model.BaseModelConfig):
        if isinstance(model_config, _value_function.ValueFunctionConfig):
            return model_config.network_config
        if isinstance(model_config, _value_function.CQLValueFunctionConfig):
            return model_config.q_network_config
        if isinstance(model_config, _value_function.IQLValueFunctionConfig):
            return model_config.q_network_config
        return None

    def _get_action_dim(self, model_config: _model.BaseModelConfig) -> int:
        network_config = self._get_critic_network_config(model_config)
        if network_config is not None and hasattr(network_config, "action_dim"):
            return network_config.action_dim
        if isinstance(model_config, pi0_config.Pi0Config):
            # Real (pre-padding) action dim. When an explicit mask is provided it is the
            # source of truth; otherwise fall back to action_dim - action_dim_offset.
            if model_config.action_dim_mask is not None:
                return int(sum(model_config.action_dim_mask))
            return model_config.action_dim - model_config.action_dim_offset
        raise ValueError(
            f"Cannot derive action_dim from model_config of type {type(model_config).__name__}"
        )

    def _create_model_transforms(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        if not self.critic_mode:
            base_transforms = ModelTransformFactory(default_prompt = None)(model_config)
            return _transforms.Group(
                inputs = (
                    DecodeRoboCoinPromptBytes(),
                    *base_transforms.inputs,
                ),
                outputs = base_transforms.outputs,
            )

        tokenizer = self._get_critic_tokenizer(model_config)
        transforms: list[_transforms.DataTransformFn] = []
        if self.replace_boundary_actions:
            transforms.append(_transforms.ReplaceMaskedActions(use_quantile_norm = self.use_quantile_norm))
        if tokenizer is not None:
            if self.subtask_prompt_mode == "all_subtasks_predict_current_subtask":
                tokenize_transform: _transforms.DataTransformFn = _transforms.TokenizeRoboCoinSubtaskPrompt(
                    tokenizer = tokenizer,
                    prefix_text = robocoin_rlds_dataset.ALL_SUBTASKS_HANG_PROMPT,
                )
            elif self.subtask_prompt_mode == "task_description_predict_current_subtask":
                tokenize_transform = _transforms.TokenizeRoboCoinSubtaskPrompt(
                    tokenizer = tokenizer,
                )
            else:
                tokenize_transform = _transforms.TokenizePrompt(tokenizer)
            # Image resizing is handled inside BaseRldsDataset before TF batching.
            transforms.extend(
                [
                    DecodeRoboCoinPromptBytes(),
                    tokenize_transform,
                ]
            )
        return _transforms.Group(inputs = transforms, outputs = [])

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.datasets:
            raise ValueError("RoboCoinRldsDataConfig requires at least one RLDS dataset.")

        asset_id = self.assets.asset_id or self.datasets[0].name
        norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)

        data_transforms_inputs: list[_transforms.DataTransformFn] = []
        data_transforms_outputs: list[_transforms.DataTransformFn] = []
        if self.use_chunk_wise_delta:
            action_dim = self._get_action_dim(model_config)
            assert self.state_dim == action_dim, (
                f"chunk-wise delta requires state_dim == action_dim, "
                f"got state_dim={self.state_dim}, action_dim={action_dim}"
            )
            # 14D EEF layout: left xyz (0-2), left rpy (3-5), left gripper (6),
            # right xyz (7-9), right rpy (10-12), right gripper (13).
            # Mask is True for non-gripper dims; the rpy slots are additionally overridden
            # by rpy_index_start so they use relative-rotation composition.
            delta_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms_inputs.append(
                _transforms.DeltaActions(mask = delta_mask, rpy_index_start = (3, 10))
            )
            data_transforms_outputs.append(
                _transforms.AbsoluteActions(mask = delta_mask, rpy_index_start = (3, 10))
            )

        return DataConfig(
            repo_id = self.repo_id,
            asset_id = asset_id,
            norm_stats = norm_stats,
            repack_transforms = _transforms.Group(inputs = []),
            data_transforms = _transforms.Group(inputs = data_transforms_inputs, outputs = data_transforms_outputs),
            model_transforms = self._create_model_transforms(model_config),
            use_quantile_norm = self.use_quantile_norm,
            critic_mode = self.critic_mode,
            discount = self.discount,
            reward_scale = self.reward_scale,
            reward_bias = self.reward_bias,
            rlds_data_dir = self.rlds_data_dir,
            val_dataset_dir = self.val_dataset_dir,
            rlds_dataset_class = "robocoin",
            datasets = self.datasets,
            robocoin_use_eef = self.use_eef,
            val_split = self.val_split,
            clip_normalized_bounds = self._create_clip_normalized_bounds(),
            counterfactual_action_store_dir = self.counterfactual_action_store_dir,
            max_num_demos = self.max_num_demos,
            rlds_kwargs = {
                "td_n": self.td_n,
                "filter_n": self.filter_n,
                "mask_50fps": self.mask_50fps,
                "mask_boundary_actions": self.mask_boundary_actions or self.replace_boundary_actions,
                "variable_horizon": self.variable_horizon,
                "lower_action_horizon": self.lower_action_horizon,
                "use_chunk_wise_delta": self.use_chunk_wise_delta,
                "shuffle_buffer_size": self.shuffle_buffer_size,
                "num_parallel_reads": self.num_parallel_reads,
                "num_parallel_calls": self.num_parallel_calls,
                "image_size": self.image_size,
                "state_dim": self.state_dim,
                "subtask_prompt_mode": self.subtask_prompt_mode,
            },
        )


@dataclasses.dataclass(frozen=True)
class Hdf5RldsDataConfig(DataConfigFactory):
    """Data config for HDF5-sourced datasets (e.g. ``real_hang``).

    Standalone sibling of ``RoboCoinRldsDataConfig`` — routes to the corresponding dataset
    via ``rlds_dataset_class = "hdf5"``. The real_hang assets happen to ship in the
    RoboCOIN multi-embodiment norm-stat format, so ``_load_norm_stats`` delegates
    to the RoboCOIN implementation (and the helpers it transitively reads off
    ``self``) without otherwise pulling in the RoboCOIN class hierarchy.
    """

    repo_id: str = "real_shirt_hang"
    assets: AssetsConfig = dataclasses.field(default_factory = AssetsConfig)

    # RLDS dataset loading
    rlds_data_dir: str = "gs://saksham-euw4/hdf5/real_hang"
    val_dataset_dir: str | None = None
    datasets: Sequence[rlds_dataset.RLDSDataset] = (
        rlds_dataset.RLDSDataset(name = "real_hang", version = "1.0.0", weight = 1.0),
    )
    val_split: str = "val"
    counterfactual_action_store_dir: str | None = None
    max_num_demos: int | None = None
    shuffle_buffer_size: int = 250_000
    num_parallel_reads: int = 8
    num_parallel_calls: int = 8

    # Image and model
    image_size: tuple[int, int] = (224, 224)
    max_token_len: int = 48

    # RL training
    discount: float = 0.99
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    critic_mode: bool = True

    # Data pipeline options
    td_n: int | None = None
    use_eef: bool = False
    use_quantile_norm: bool = False
    filter_n: int | None = None
    filter_intervention: bool = False
    # When not None, drop all frames whose int32 `repo_index` is not in this list.
    filter_repo_index: tuple[int, ...] | None = None
    mask_boundary_actions: bool = True
    replace_boundary_actions: bool = False
    variable_horizon: bool = False
    use_chunk_wise_delta: bool = False
    state_dim: int = 16
    prompt_mode: hdf5_rlds_dataset.PromptMode = "subtask"
    subsample: bool = False

    def __post_init__(self) -> None:
        if self.mask_boundary_actions and self.replace_boundary_actions:
            raise ValueError("At most one of mask_boundary_actions and replace_boundary_actions can be True.")
        if self.variable_horizon and self.mask_boundary_actions:
            raise ValueError("variable_horizon=True requires mask_boundary_actions=False")
        # State-dim invariant: (state_dim=16, use_eef=False) or (state_dim=14, use_eef=True).
        if not ((self.state_dim == 16 and not self.use_eef) or (self.state_dim == 14 and self.use_eef)):
            raise ValueError(
                "Hdf5RldsDataConfig requires (state_dim=16, use_eef=False) or "
                f"(state_dim=14, use_eef=True); got state_dim={self.state_dim}, use_eef={self.use_eef}"
            )

    # No _load_norm_stats override — the base DataConfigFactory loader handles
    # the standard ``{"norm_stats": {...}}`` files compute_norm_stats writes.

    @staticmethod
    def _broadcast_gripper_stats(abs_slice, ref_slice):
        return RoboCoinRldsDataConfig._broadcast_gripper_stats(abs_slice, ref_slice)

    @staticmethod
    def _combine_eef_and_gripper_stats(eef_arr, joint_arr):
        return RoboCoinRldsDataConfig._combine_eef_and_gripper_stats(eef_arr, joint_arr)

    @staticmethod
    def _pad_14_to_16(x, fill_value):
        return RoboCoinRldsDataConfig._pad_14_to_16(x, fill_value)

    def _create_clip_normalized_bounds(self) -> dict[str, tuple[float, float]]:
        clip_bound = 1.25 if self.use_quantile_norm else 5.0
        return {
            "state": (-clip_bound, clip_bound),
            "actions": (-clip_bound, clip_bound),
            "next_state": (-clip_bound, clip_bound),
            "next_actions": (-clip_bound, clip_bound),
            "counterfactual_actions": (-clip_bound, clip_bound),
            "counterfactual_next_actions": (-clip_bound, clip_bound),
        }

    def _get_critic_network_config(self, model_config: _model.BaseModelConfig):
        if isinstance(model_config, _value_function.ValueFunctionConfig):
            return model_config.network_config
        if isinstance(model_config, _value_function.CQLValueFunctionConfig):
            return model_config.q_network_config
        if isinstance(model_config, _value_function.IQLValueFunctionConfig):
            return model_config.q_network_config
        return None

    def _get_critic_tokenizer(
        self, model_config: _model.BaseModelConfig
    ) -> _tokenizer.PaligemmaTokenizer | _tokenizer.Gemma3Tokenizer | _tokenizer.Gemma4Tokenizer | None:
        network_config = self._get_critic_network_config(model_config)
        if isinstance(network_config, _paligemma_network.PaliGemmaNetworkConfig):
            return network_config.get_tokenizer(max_len = self.max_token_len)
        return None

    def _get_action_dim(self, model_config: _model.BaseModelConfig) -> int:
        network_config = self._get_critic_network_config(model_config)
        if network_config is not None and hasattr(network_config, "action_dim"):
            return network_config.action_dim
        if isinstance(model_config, pi0_config.Pi0Config):
            if model_config.action_dim_mask is not None:
                return int(sum(model_config.action_dim_mask))
            return model_config.action_dim - model_config.action_dim_offset
        raise ValueError(
            f"Cannot derive action_dim from model_config of type {type(model_config).__name__}"
        )

    def _create_model_transforms(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        # HDF5 only has a single subtask per episode (no all_subtasks variant), but the
        # `task_description_predict_current_subtask` mode routes through the same
        # TokenizeRoboCoinSubtaskPrompt path as RoboCOIN.
        if not self.critic_mode:
            base_transforms = ModelTransformFactory(default_prompt = None)(model_config)
            return _transforms.Group(
                inputs = (
                    DecodeRoboCoinPromptBytes(),
                    *base_transforms.inputs,
                ),
                outputs = base_transforms.outputs,
            )

        tokenizer = self._get_critic_tokenizer(model_config)
        transforms: list[_transforms.DataTransformFn] = []
        if self.replace_boundary_actions:
            transforms.append(_transforms.ReplaceMaskedActions(use_quantile_norm = self.use_quantile_norm))
        if tokenizer is not None:
            if self.prompt_mode == "task_description_predict_current_subtask":
                tokenize_transform: _transforms.DataTransformFn = _transforms.TokenizeRoboCoinSubtaskPrompt(
                    tokenizer = tokenizer,
                )
            else:
                tokenize_transform = _transforms.TokenizePrompt(tokenizer)
            transforms.extend(
                [
                    DecodeRoboCoinPromptBytes(),
                    tokenize_transform,
                ]
            )
        return _transforms.Group(inputs = transforms, outputs = [])

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.datasets:
            raise ValueError("Hdf5RldsDataConfig requires at least one RLDS dataset.")

        asset_id = self.assets.asset_id or self.datasets[0].name
        norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)

        # Mirror RLDSRoboCasaDataConfig: when compute_norm_stats produced the standard
        # openpi format (no RoboCOIN-format conversion), route action_diff stats into
        # `actions` for use_chunk_wise_delta runtimes and replicate state/actions into
        # next_state/next_actions. The RoboCOIN loader already does this internally, so
        # the action_diff routing only fires when the absolute-action stats are still in
        # `actions` (i.e., for standard-format files) and an action_diff entry is present.
        if norm_stats is not None:
            if self.use_chunk_wise_delta and "action_diff" in norm_stats:
                norm_stats["actions"] = _slice_action_diff_norm_stats(
                    norm_stats["action_diff"], model_config.action_horizon,
                    subsample = self.subsample,
                )
            if "state" in norm_stats and "next_state" not in norm_stats:
                norm_stats["next_state"] = norm_stats["state"]
            if "actions" in norm_stats and "next_actions" not in norm_stats:
                norm_stats["next_actions"] = norm_stats["actions"]
            # Cached counterfactual actions are absolute+unnormalized on disk; DeltaActions
            # converts them to chunk-wise-delta and these aliases route them through
            # Normalize+Clip alongside (next_)actions. Without them the cf_* fields skip
            # Normalize entirely (apply_tree strict=False is a no-op for missing keys),
            # leaving them in raw delta units while the critic was trained on normalized
            # ones. Mirrors RoboCoinRldsDataConfig._load_norm_stats:1068-1069.
            if "actions" in norm_stats and "counterfactual_actions" not in norm_stats:
                norm_stats["counterfactual_actions"] = norm_stats["actions"]
                norm_stats["counterfactual_next_actions"] = norm_stats["actions"]

        data_transforms_inputs: list[_transforms.DataTransformFn] = []
        data_transforms_outputs: list[_transforms.DataTransformFn] = []
        if self.use_chunk_wise_delta:
            action_dim = self._get_action_dim(model_config)
            assert self.state_dim == action_dim, (
                f"chunk-wise delta requires state_dim == action_dim, "
                f"got state_dim={self.state_dim}, action_dim={action_dim}"
            )
            delta_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms_inputs.append(
                _transforms.DeltaActions(mask = delta_mask, rpy_index_start = (3, 10))
            )
            data_transforms_outputs.append(
                _transforms.AbsoluteActions(mask = delta_mask, rpy_index_start = (3, 10))
            )

        return DataConfig(
            repo_id = self.repo_id,
            asset_id = asset_id,
            norm_stats = norm_stats,
            repack_transforms = _transforms.Group(inputs = []),
            data_transforms = _transforms.Group(inputs = data_transforms_inputs, outputs = data_transforms_outputs),
            model_transforms = self._create_model_transforms(model_config),
            use_quantile_norm = self.use_quantile_norm,
            critic_mode = self.critic_mode,
            discount = self.discount,
            reward_scale = self.reward_scale,
            reward_bias = self.reward_bias,
            rlds_data_dir = self.rlds_data_dir,
            val_dataset_dir = self.val_dataset_dir,
            rlds_dataset_class = "hdf5",
            datasets = self.datasets,
            robocoin_use_eef = self.use_eef,
            val_split = self.val_split,
            clip_normalized_bounds = self._create_clip_normalized_bounds(),
            counterfactual_action_store_dir = self.counterfactual_action_store_dir,
            max_num_demos = self.max_num_demos,
            rlds_kwargs = {
                "td_n": self.td_n,
                "filter_n": self.filter_n,
                "filter_intervention": self.filter_intervention,
                "filter_repo_index": self.filter_repo_index,
                "mask_boundary_actions": self.mask_boundary_actions or self.replace_boundary_actions,
                "variable_horizon": self.variable_horizon,
                "use_chunk_wise_delta": self.use_chunk_wise_delta,
                "shuffle_buffer_size": self.shuffle_buffer_size,
                "num_parallel_reads": self.num_parallel_reads,
                "num_parallel_calls": self.num_parallel_calls,
                "image_size": self.image_size,
                "state_dim": self.state_dim,
                "prompt_mode": self.prompt_mode,
                "subsample": self.subsample,
            },
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRldsDataConfig(DataConfigFactory):
    """Data config for LeRobot-built RLDS datasets (e.g. ``realworld_xarm_packing``).

    Routes to ``LeRobotRldsDataset`` via ``rlds_dataset_class = "lerobot"``. State and
    action are already 14D EEF, so there is no ``use_eef`` / ``state_dim`` knob.
    Supports both behavior cloning (``critic_mode=False``) and value-function training
    (``critic_mode=True``).
    """

    repo_id: str = "realworld_xarm_packing"
    assets: AssetsConfig = dataclasses.field(default_factory = AssetsConfig)

    rlds_data_dir: str = "gs://saksham-euw4/datasets/realworld_xarm_packing"
    val_dataset_dir: str | None = None
    datasets: Sequence[rlds_dataset.RLDSDataset] = (
        rlds_dataset.RLDSDataset(name = "realworld_xarm_packing", version = "1.0.0", weight = 1.0),
    )
    val_split: str = "val"
    shuffle_buffer_size: int = 250_000
    num_parallel_reads: int = 8
    num_parallel_calls: int = 8

    image_size: tuple[int, int] = (224, 224)
    max_token_len: int = 48

    use_quantile_norm: bool = False
    use_chunk_wise_delta: bool = False
    filter_n: int | None = None
    prompt_mode: lerobot_rlds_dataset.PromptMode = "subtask"

    # RL / value-function training
    critic_mode: bool = False
    discount: float = 0.99
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    td_n: int | None = None
    mask_boundary_actions: bool = True
    subsample: bool = False
    counterfactual_action_store_dir: str | None = None

    def _create_clip_normalized_bounds(self) -> dict[str, tuple[float, float]]:
        clip_bound = 1.25 if self.use_quantile_norm else 5.0
        bounds = {
            "state": (-clip_bound, clip_bound),
            "actions": (-clip_bound, clip_bound),
        }
        if self.critic_mode:
            bounds.update({
                "next_state": (-clip_bound, clip_bound),
                "next_actions": (-clip_bound, clip_bound),
                "counterfactual_actions": (-clip_bound, clip_bound),
                "counterfactual_next_actions": (-clip_bound, clip_bound),
            })
        return bounds

    def _get_critic_network_config(self, model_config: _model.BaseModelConfig):
        if isinstance(model_config, _value_function.ValueFunctionConfig):
            return model_config.network_config
        if isinstance(model_config, _value_function.CQLValueFunctionConfig):
            return model_config.q_network_config
        if isinstance(model_config, _value_function.IQLValueFunctionConfig):
            return model_config.q_network_config
        return None

    def _get_critic_tokenizer(
        self, model_config: _model.BaseModelConfig
    ) -> _tokenizer.PaligemmaTokenizer | _tokenizer.Gemma3Tokenizer | _tokenizer.Gemma4Tokenizer | None:
        network_config = self._get_critic_network_config(model_config)
        if isinstance(network_config, _paligemma_network.PaliGemmaNetworkConfig):
            return network_config.get_tokenizer(max_len = self.max_token_len)
        return None

    def _get_action_dim(self, model_config: _model.BaseModelConfig) -> int:
        network_config = self._get_critic_network_config(model_config)
        if network_config is not None and hasattr(network_config, "action_dim"):
            return network_config.action_dim
        if isinstance(model_config, pi0_config.Pi0Config):
            if model_config.action_dim_mask is not None:
                return int(sum(model_config.action_dim_mask))
            return model_config.action_dim - model_config.action_dim_offset
        raise ValueError(
            f"Cannot derive action_dim from model_config of type {type(model_config).__name__}"
        )

    def _create_model_transforms(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        if not self.critic_mode:
            base_transforms = ModelTransformFactory(default_prompt = None)(model_config)
            return _transforms.Group(
                inputs = (
                    DecodeRoboCoinPromptBytes(),
                    *base_transforms.inputs,
                ),
                outputs = base_transforms.outputs,
            )

        tokenizer = self._get_critic_tokenizer(model_config)
        transforms: list[_transforms.DataTransformFn] = []
        if tokenizer is not None:
            if self.prompt_mode == "task_description_predict_current_subtask":
                tokenize_transform: _transforms.DataTransformFn = _transforms.TokenizeRoboCoinSubtaskPrompt(
                    tokenizer = tokenizer,
                )
            else:
                tokenize_transform = _transforms.TokenizePrompt(tokenizer)
            transforms.extend(
                [
                    DecodeRoboCoinPromptBytes(),
                    tokenize_transform,
                ]
            )
        return _transforms.Group(inputs = transforms, outputs = [])

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.datasets:
            raise ValueError("LeRobotRldsDataConfig requires at least one RLDS dataset.")

        asset_id = self.assets.asset_id or self.datasets[0].name
        norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)

        if norm_stats is not None:
            if self.use_chunk_wise_delta and "action_diff" in norm_stats:
                norm_stats["actions"] = _slice_action_diff_norm_stats(
                    norm_stats["action_diff"], model_config.action_horizon,
                    subsample = self.subsample,
                )
            # Critic mode reuses the state/actions stats for the next_* and cached
            # counterfactual fields so they go through Normalize+Clip identically.
            if self.critic_mode:
                if "state" in norm_stats and "next_state" not in norm_stats:
                    norm_stats["next_state"] = norm_stats["state"]
                if "actions" in norm_stats and "next_actions" not in norm_stats:
                    norm_stats["next_actions"] = norm_stats["actions"]
                if "actions" in norm_stats and "counterfactual_actions" not in norm_stats:
                    norm_stats["counterfactual_actions"] = norm_stats["actions"]
                    norm_stats["counterfactual_next_actions"] = norm_stats["actions"]

        data_transforms_inputs: list[_transforms.DataTransformFn] = []
        data_transforms_outputs: list[_transforms.DataTransformFn] = []
        if self.use_chunk_wise_delta:
            action_dim = self._get_action_dim(model_config)
            assert action_dim == 14, f"LeRobotRldsDataConfig expects 14D EEF actions, got {action_dim}"
            delta_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms_inputs.append(
                _transforms.DeltaActions(mask = delta_mask, rpy_index_start = (3, 10))
            )
            data_transforms_outputs.append(
                _transforms.AbsoluteActions(mask = delta_mask, rpy_index_start = (3, 10))
            )

        return DataConfig(
            repo_id = self.repo_id,
            asset_id = asset_id,
            norm_stats = norm_stats,
            repack_transforms = _transforms.Group(inputs = []),
            data_transforms = _transforms.Group(inputs = data_transforms_inputs, outputs = data_transforms_outputs),
            model_transforms = self._create_model_transforms(model_config),
            use_quantile_norm = self.use_quantile_norm,
            critic_mode = self.critic_mode,
            discount = self.discount,
            reward_scale = self.reward_scale,
            reward_bias = self.reward_bias,
            rlds_data_dir = self.rlds_data_dir,
            val_dataset_dir = self.val_dataset_dir,
            rlds_dataset_class = "lerobot",
            datasets = self.datasets,
            val_split = self.val_split,
            clip_normalized_bounds = self._create_clip_normalized_bounds(),
            counterfactual_action_store_dir = self.counterfactual_action_store_dir,
            rlds_kwargs = {
                "td_n": self.td_n,
                "filter_n": self.filter_n,
                "mask_boundary_actions": self.mask_boundary_actions,
                "prompt_mode": self.prompt_mode,
                "subsample": self.subsample,
                "shuffle_buffer_size": self.shuffle_buffer_size,
                "num_parallel_reads": self.num_parallel_reads,
                "num_parallel_calls": self.num_parallel_calls,
                "image_size": self.image_size,
            },
        )


def _slice_action_diff_norm_stats(
    stats: _transforms.NormStats, action_horizon: int, subsample: bool = False,
) -> _transforms.NormStats:
    """Slice the leading time axis of a 2-D `(H, D)` NormStats to `(action_horizon, D)`.

    `compute_norm_stats` writes `action_diff` stats at a fixed length covering the
    longest action_horizon any consumer uses; the runtime config slices down to its
    own action_horizon when routing `action_diff` into the `actions` key.

    When `subsample=True` (matches `Hdf5RldsDataset(subsample=True)`), strides the
    stored stats `[1::2]` so the kept slots align with the half-cadence action
    chunks the model actually sees. If the resulting array is shorter than
    `action_horizon`, pads the time axis with zeros to match.
    """
    import numpy as np

    # matches subsample=True in Hdf5RldsDataset processing
    def _stride(arr):
        return None if arr is None else arr[1::2]

    if subsample:
        stats = _transforms.NormStats(
            mean = _stride(stats.mean),
            std = _stride(stats.std),
            q01 = _stride(stats.q01),
            q99 = _stride(stats.q99),
        )

    def _slice_and_pad(arr):
        if arr is None:
            return None
        sliced = arr[:action_horizon]
        if sliced.shape[0] < action_horizon:
            pad_len = action_horizon - sliced.shape[0]
            pad = np.zeros((pad_len, *sliced.shape[1:]), dtype = sliced.dtype)
            sliced = np.concatenate([sliced, pad], axis = 0)
        return sliced

    return _transforms.NormStats(
        mean = _slice_and_pad(stats.mean),
        std = _slice_and_pad(stats.std),
        q01 = _slice_and_pad(stats.q01),
        q99 = _slice_and_pad(stats.q99),
    )


@dataclasses.dataclass(frozen=True)
class RLDSRoboCasaDataConfig(DataConfigFactory):
    """Config for training on RoboCasa using RLDS data format.

    Expects data converted by scripts/convert_robocasa_to_rlds.py with flat
    {split}__{category}__{task} naming convention. Data dir points to the root
    directory containing all dataset subdirectories.

    Camera mapping:
        robot0_agentview_left  -> observation/image
        robot0_agentview_right -> observation/image_right
        robot0_eye_in_hand     -> observation/wrist_image
    """

    repo_id: str | None = None
    rlds_data_dir: str = "gs://saksham-euw4/robocasa"
    datasets: Sequence[rlds_dataset.RLDSDataset] = (
        rlds_dataset.RLDSDataset(name = "target__composite__load_dishwasher", version = "1.0.0", weight = 1.0),
    )

    critic_mode: bool = False
    image_size: int = 224
    max_num_demos: int | None = None

    # If True, map RoboCasa's single-arm 12D action / 13D state into the RoboCOIN
    # bimanual-EEF 14D layout (right-arm slot only, left-arm zeros, base/control_mode
    # dropped) and use the bimanual 3-camera layout (right_wrist masked).
    # Actions stay absolute base-frame poses (no chunk-wise delta) unless
    # use_chunk_wise_delta=True.
    bimanual_eef_layout: bool = False

    # When True under bimanual_eef_layout, route all three RoboCasa cameras into the
    # bimanual image slots instead of masking the third. Layout becomes
    # (left/top → base_0_rgb, right/top → left_wrist_0_rgb, wrist_camera → right_wrist_0_rgb)
    # with all three image_mask entries True. No-op under bimanual_eef_layout=False
    # (that path already routes all three cameras).
    use_all_cameras: bool = False

    # RoboCOIN-style chunk-wise-delta normalization (mirrors `RoboCoinRldsDataConfig.use_chunk_wise_delta`).
    # When True, DeltaActions runs as input transform and AbsoluteActions as output transform,
    # with `rpy_index_start` covering the right-arm euler-xyz rotation block. Default False —
    # the model trains on absolute eef pose targets.
    use_chunk_wise_delta: bool = False

    # None = inherit `create_base_config`'s default (True for non-PI0 model_type).
    use_quantile_norm: bool | None = None

    # FPS interpolation. When `interpolation_config` is set, RoboCasaRldsDataset resamples
    # each trajectory from `native_fps` to `interpolation_config.target_fps`. Required to
    # match the 30-fps convention of the RoboCOIN-pretrained value head / π-0.5 policy.
    interpolation_config: state_action_spaces.InterpolationConfig | None = None
    native_fps: float | None = None

    # Mirrors RoboCoinRldsDataConfig: at most one True. mask=True emits
    # tail-False action_mask; replace=True also runs ReplaceMaskedActions.
    mask_boundary_actions: bool = True
    replace_boundary_actions: bool = False

    shuffle_buffer_size: int = 25_000

    # RL training parameters
    discount: float = 0.99
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # n-step horizon used for TD targets, in canonical 50Hz units. Mirrors RoboCOIN:
    # at 30fps the dataset uses 3*td_n/5 native steps.
    td_n: int = 50

    # "subtask" (default) runs the composite subtask tracker for composite datasets;
    # "task_description" bypasses the tracker so the raw per-step language_instruction
    # is the prompt for every dataset (atomic and composite alike).
    prompt_mode: robocasa_rlds_dataset.PromptMode = "subtask"

    def __post_init__(self) -> None:
        if self.mask_boundary_actions and self.replace_boundary_actions:
            raise ValueError(
                "At most one of mask_boundary_actions and replace_boundary_actions can be True."
            )
        if self.td_n % 5 != 0:
            raise ValueError(f"td_n must be a multiple of 5, got {self.td_n}")

    # Mirror RoboCoinRldsDataConfig for the val plotting path.
    def _get_critic_network_config(self, model_config: _model.BaseModelConfig):
        if isinstance(model_config, _value_function.ValueFunctionConfig):
            return model_config.network_config
        if isinstance(model_config, _value_function.IQLValueFunctionConfig):
            return model_config.q_network_config
        if isinstance(model_config, _value_function.CQLValueFunctionConfig):
            return model_config.q_network_config
        return None

    def _get_critic_tokenizer(
        self, model_config: _model.BaseModelConfig
    ) -> _tokenizer.PaligemmaTokenizer | _tokenizer.Gemma3Tokenizer | _tokenizer.Gemma4Tokenizer | None:
        network_config = self._get_critic_network_config(model_config)
        if isinstance(network_config, _paligemma_network.PaliGemmaNetworkConfig):
            return network_config.get_tokenizer()
        return None

    def _create_clip_normalized_bounds(self) -> dict[str, tuple[float, float]]:
        clip_bound = 1.25 if bool(self.use_quantile_norm) else 5.0
        return {
            "state": (-clip_bound, clip_bound),
            "actions": (-clip_bound, clip_bound),
            "next_state": (-clip_bound, clip_bound),
            "next_actions": (-clip_bound, clip_bound),
        }

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.interpolation_config is not None:
            ic = self.interpolation_config
            if ic.target_fps != 30.0:
                raise ValueError(
                    f"RLDSRoboCasaDataConfig only supports target_fps=30.0, got {ic.target_fps}"
                )
            if self.native_fps is None:
                raise ValueError("interpolation_config requires native_fps to be set.")
            expected_horizon = round(ic.action_horizon_seconds * ic.target_fps)
            # Pi0Config exposes action_horizon directly; value-function configs leave it as
            # None and the chunk size comes from TrainConfig.action_horizon (which the
            # FineTuneConfig overrides for fine-tunes). Only enforce when the model carries
            # action_horizon explicitly.
            model_action_horizon = getattr(model_config, "action_horizon", None)
            if model_action_horizon is not None:
                assert model_action_horizon >= expected_horizon, (
                    f"model.action_horizon ({model_action_horizon}) must be >= "
                    f"action_horizon_seconds * target_fps ({expected_horizon}) when "
                    f"interpolation_config is set."
                )
        # Resolve action_dim: Pi0Config exposes it directly; value-function configs
        # (SARSAValueFunctionConfig, etc.) keep it under network_config.
        if hasattr(model_config, "action_dim"):
            model_action_dim = model_config.action_dim
        elif hasattr(model_config, "network_config") and hasattr(model_config.network_config, "action_dim"):
            model_action_dim = model_config.network_config.action_dim
        else:
            raise ValueError(f"Cannot resolve action_dim from {type(model_config).__name__}")

        # Repack: RoboCasaRldsDataset.trajectory_transforms already remaps cameras to
        # cam_0/cam_1/cam_2 (left/right/eye_in_hand) for joint-pipeline compatibility,
        # so we re-key from there into the policy-side image / image_right / wrist_image
        # slots that RoboCasaInputs / RoboCasaBimanualEEFInputs expect.
        repack_mapping = {
            "observation/image": "observation/cam_0",
            "observation/image_right": "observation/cam_1",
            "observation/wrist_image": "observation/cam_2",
            "observation/state": "observation/state",
            "actions": "actions",
            "prompt": "prompt",
            "_frame_index": "_frame_index",
            "_traj_index": "_traj_index",
            "repo_id": "repo_id",
        }

        rlds_kwargs: dict[str, Any] = {
            "image_size": (self.image_size, self.image_size),
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "mask_boundary_actions": self.mask_boundary_actions or self.replace_boundary_actions,
            "prompt_mode": self.prompt_mode,
        }
        if self.interpolation_config is not None:
            # Interpolation runs INSIDE trajectory_transforms, before any bimanual-EEF
            # remapping, so the specs match RoboCasa-native shapes (12D action, 13D state).
            rlds_kwargs["interpolation_config"] = self.interpolation_config
            rlds_kwargs["action_space_spec"] = state_action_spaces.ROBOCASA_NATIVE_ACTION_SPEC
            rlds_kwargs["state_space_spec"] = state_action_spaces.ROBOCASA_NATIVE_STATE_SPEC
            rlds_kwargs["native_fps"] = self.native_fps
        repack_mapping["action_mask"] = "action_mask"

        if self.critic_mode:
            rlds_kwargs["td_n"] = self.td_n

            repack_mapping.update(
                {
                    "next_observation/image": "next_observation/cam_0",
                    "next_observation/image_right": "next_observation/cam_1",
                    "next_observation/wrist_image": "next_observation/cam_2",
                    "next_observation/state": "next_observation/state",
                    "next_actions": "next_actions",
                    "reward": "reward",
                    "mc_return": "mc_return",
                    "termination": "termination",
                    "truncation": "truncation",
                    "td_discount": "td_discount",
                    "next_action_mask": "next_action_mask",
                    "fps": "fps",
                    "steps_to_subtask_end": "steps_to_subtask_end",
                }
            )
            repack_transform = _transforms.Group(inputs = [_transforms.RepackTransform(repack_mapping)])
            data_transforms_inputs: list[_transforms.DataTransformFn]
            data_transforms_outputs: list[_transforms.DataTransformFn]
            if self.bimanual_eef_layout:
                data_transforms_inputs = [
                    robocasa_policy.RoboCasaBimanualEEFInputs(
                        action_dim = model_action_dim, use_all_cameras = self.use_all_cameras,
                    )
                ]
                data_transforms_outputs = [
                    robocasa_policy.RoboCasaBimanualEEFOutputs(),
                    robocasa_policy.RoboCasaEefRotEulerXYZToAxisAngle(),
                ]
            else:
                data_transforms_inputs = [robocasa_policy.RoboCasaInputs(action_dim = model_action_dim)]
                data_transforms_outputs = [robocasa_policy.RoboCasaOutputs()]

            if self.use_chunk_wise_delta:
                if not self.bimanual_eef_layout:
                    raise ValueError(
                        "use_chunk_wise_delta=True is only supported under bimanual_eef_layout=True "
                        "(rpy_index_start hardcoded for the right-arm slot at 3)."
                    )
                delta_mask = _transforms.make_bool_mask(6, -8)
                # DeltaActions runs AFTER bimanual_input_transform so it sees the 14D arm-first
                # bimanual layout (eef_pos:[0:3], eef_rot:[3:6], gripper:[6], padding:[7:14]).
                data_transforms_inputs.append(
                    _transforms.DeltaActions(mask = delta_mask, rpy_index_start = (3,))
                )
                # AbsoluteActions undoes chunk-wise-delta BEFORE the bimanual output transform
                # slices the arm dims out for RoboCasa-native action assembly.
                data_transforms_outputs.insert(
                    0, _transforms.AbsoluteActions(mask = delta_mask, rpy_index_start = (3,))
                )

            data_transforms = _transforms.Group(
                inputs = data_transforms_inputs, outputs = data_transforms_outputs,
            )

            # Build model_transforms: ResizeImages, optional ReplaceMaskedActions,
            # and (for paligemma value backbones) decode + tokenize.
            model_inputs: list[_transforms.DataTransformFn] = [
                _transforms.ResizeImages(self.image_size, self.image_size),
            ]
            if self.replace_boundary_actions:
                model_inputs.append(_transforms.ReplaceMaskedActions(use_quantile_norm = bool(self.use_quantile_norm)))
            critic_network_config = None
            if isinstance(model_config, _value_function.ValueFunctionConfig):
                critic_network_config = model_config.network_config
            elif isinstance(model_config, _value_function.IQLValueFunctionConfig):
                critic_network_config = model_config.q_network_config
            elif isinstance(model_config, _value_function.CQLValueFunctionConfig):
                critic_network_config = model_config.q_network_config
            if isinstance(critic_network_config, _paligemma_network.PaliGemmaNetworkConfig):
                tokenizer = critic_network_config.get_tokenizer()
                model_inputs.append(DecodeRoboCoinPromptBytes())
                model_inputs.append(_transforms.TokenizePrompt(tokenizer))
            model_transforms = _transforms.Group(inputs = model_inputs)

            asset_id = self.assets.asset_id
            norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)
            if norm_stats is not None:
                # compute_norm_stats writes both 'actions' (1-D, action[0] absolute) and
                # 'action_diff' (2-D, per-step DeltaActions output of length ACTION_DIFF_HORIZON).
                # Route the runtime 'actions' key to whichever flavor matches the data
                # pipeline: when use_chunk_wise_delta=True the data has DeltaActions applied,
                # so we use the per-step delta stats sliced down to action_horizon.
                if self.use_chunk_wise_delta:
                    norm_stats["actions"] = _slice_action_diff_norm_stats(
                        norm_stats["action_diff"], model_config.action_horizon,
                    )
                if "state" in norm_stats:
                    norm_stats["next_state"] = norm_stats["state"]
                if "actions" in norm_stats:
                    norm_stats["next_actions"] = norm_stats["actions"]

            return DataConfig(
                repo_id = "robocasa",
                asset_id = asset_id,
                norm_stats = norm_stats,
                repack_transforms = repack_transform,
                data_transforms = data_transforms,
                model_transforms = model_transforms,
                use_quantile_norm = bool(self.use_quantile_norm),
                clip_normalized_bounds = self._create_clip_normalized_bounds(),
                critic_mode = True,
                discount = self.discount,
                reward_scale = self.reward_scale,
                reward_bias = self.reward_bias,
                rlds_data_dir = self.rlds_data_dir,
                rlds_dataset_class = "robocasa",
                datasets = self.datasets,
                max_num_demos = self.max_num_demos,
                rlds_kwargs = rlds_kwargs,
                val_split = "train",
            )

        repack_transform = _transforms.Group(inputs = [_transforms.RepackTransform(repack_mapping)])

        sup_inputs: list[_transforms.DataTransformFn]
        sup_outputs: list[_transforms.DataTransformFn]
        if self.bimanual_eef_layout:
            sup_inputs = [
                robocasa_policy.RoboCasaBimanualEEFInputs(
                    action_dim = model_action_dim,
                    model_type = model_config.model_type,
                    use_all_cameras = self.use_all_cameras,
                )
            ]
            sup_outputs = [
                robocasa_policy.RoboCasaBimanualEEFOutputs(),
                robocasa_policy.RoboCasaEefRotEulerXYZToAxisAngle(),
            ]
        else:
            sup_inputs = [
                robocasa_policy.RoboCasaInputs(
                    action_dim = model_action_dim, model_type = model_config.model_type
                )
            ]
            sup_outputs = [robocasa_policy.RoboCasaOutputs()]

        if self.use_chunk_wise_delta:
            if not self.bimanual_eef_layout:
                raise ValueError(
                    "use_chunk_wise_delta=True is only supported under bimanual_eef_layout=True "
                    "(rpy_index_start hardcoded for the right-arm slot at 3)."
                )
            delta_mask = _transforms.make_bool_mask(6, -8)
            sup_inputs.append(
                _transforms.DeltaActions(mask = delta_mask, rpy_index_start = (3,))
            )
            sup_outputs.insert(
                0, _transforms.AbsoluteActions(mask = delta_mask, rpy_index_start = (3,))
            )

        data_transforms = _transforms.Group(inputs = sup_inputs, outputs = sup_outputs)

        # The RLDS prompt arrives as numpy bytes (TFDS string encoding); TokenizePrompt
        # only handles str / numpy 0-d arrays. Decode upfront so the rest of model_transforms
        # sees a Python str. Mirrors the RoboCOIN model_transforms pipeline (config.py:1174).
        base_model_transforms = ModelTransformFactory()(model_config)
        model_transforms = _transforms.Group(
            inputs = (DecodeRoboCoinPromptBytes(), *base_model_transforms.inputs),
            outputs = base_model_transforms.outputs,
        )

        # Load and route norm_stats so the runtime 'actions' / 'next_actions' keys point
        # at the right flavor for this pipeline. compute_norm_stats writes 'actions'
        # (1-D, action[0] absolute) and 'action_diff' (2-D, per-step DeltaActions output
        # of length ACTION_DIFF_HORIZON); when use_chunk_wise_delta=True the data
        # pipeline applies DeltaActions, so Normalize must use the delta stats sliced
        # to action_horizon. Mirrors the critic-mode branch above.
        asset_id = self.assets.asset_id
        norm_stats = self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id)
        if norm_stats is not None:
            if self.use_chunk_wise_delta:
                norm_stats["actions"] = _slice_action_diff_norm_stats(
                    norm_stats["action_diff"], model_config.action_horizon,
                )
            if "state" in norm_stats:
                norm_stats["next_state"] = norm_stats["state"]
            if "actions" in norm_stats:
                norm_stats["next_actions"] = norm_stats["actions"]

        cfg = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repo_id = "robocasa",
            repack_transforms = repack_transform,
            data_transforms = data_transforms,
            model_transforms = model_transforms,
            rlds_data_dir = self.rlds_data_dir,
            rlds_dataset_class = "robocasa",
            datasets = self.datasets,
            max_num_demos = self.max_num_demos,
            rlds_kwargs = rlds_kwargs,
            norm_stats = norm_stats,
            asset_id = asset_id,
            clip_normalized_bounds = self._create_clip_normalized_bounds(),
        )
        if self.use_quantile_norm is not None:
            cfg = dataclasses.replace(cfg, use_quantile_norm = self.use_quantile_norm)
        return cfg


@dataclasses.dataclass(frozen=True)
class EvalEnvConfig:
    """Base configuration for evaluation environments."""

    # Number of evaluation episodes per eval run
    num_eval_episodes: int = 10
    # Maximum steps per episode (0 = use environment default)
    max_episode_steps: int = 0
    # Seed for environment initialization
    seed: int = 42
    # Whether to record video during evaluation.
    record_video: bool = True


@dataclasses.dataclass(frozen=True)
class MinariEvalEnvConfig(EvalEnvConfig):
    """Evaluation environment recovered from a Minari dataset.

    If minari_dataset_id is None, will be inferred from DataConfig.minari_dataset_id.
    """

    minari_dataset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class LegacyD4RLEvalEnvConfig(EvalEnvConfig):
    """Evaluation environment created from legacy D4RL.

    If legacy_d4rl_env_name is None, will be inferred from DataConfig.legacy_d4rl_env_name.
    """

    legacy_d4rl_env_name: str | None = None


@dataclasses.dataclass(frozen=True)
class FineTuneConfig:
    """Configuration for fine-tuning or validation-only evaluation on a different dataset.

    Registered in _FINE_TUNE_CONFIGS and referenced by name from TrainConfig.fine_tune.
    When applied, overrides the base config's dataset, schedule, and checkpoint intervals.
    """

    name: str = ""

    # Dataset overrides (applied via dataclasses.replace on config.data).
    # Special keys: "data_dir" -> rlds_data_dir, "dataset_name" -> splits "name:version" into datasets tuple.
    # All other keys pass through directly.
    data_overrides: dict[str, Any] = dataclasses.field(default_factory = dict)

    # Full data-factory replacement. When set, replaces config.data entirely (e.g. to swap
    # from RoboCoinRldsDataConfig to RLDSRoboCasaDataConfig). data_overrides is ignored
    # if this is non-None.
    data_factory: "DataConfigFactory | None" = None

    # Validation overrides
    include_repos: tuple[str, ...] | None = None
    validation_cache_dir: str | None = None
    num_val_trajectories: int | None = None

    # Training schedule overrides (num_train_steps is relative to pretrained step)
    num_train_steps: int | None = None
    lr_schedule: _optimizer.LRScheduleConfig | None = None

    # If True, skip training: load checkpoint, run validation, exit.
    val_only: bool = False

    # Interval / checkpoint overrides
    save_interval: int | None = None
    plot_interval: int | None = None
    keep_period: int | None = None
    log_interval: int | None = None

    action_horizon: int | None = None

    # Model config overrides (applied via dataclasses.replace on config.model)
    model_overrides: dict[str, Any] = dataclasses.field(default_factory = dict)

    # Policy config overrides (applied via dataclasses.replace on config.policy).
    # E.g. ``{"action_horizon": 60}`` to bump a Best-of-N wrapper's expected chunk
    # length to match a fine-tune-time data pipeline that differs from the
    # pretrain horizon.
    policy_overrides: dict[str, Any] = dataclasses.field(default_factory = dict)

    # If true, will overwrite the fine-tune checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume fine-tuning from the last fine-tune checkpoint.
    resume: bool = False

    # Fields that map 1:1 from FineTuneConfig to TrainConfig for apply_overrides.
    _TRAIN_CONFIG_FIELDS: ClassVar[tuple[str, ...]] = (
        "save_interval", "plot_interval", "keep_period", "log_interval",
        "include_repos", "validation_cache_dir", "num_val_trajectories",
        "action_horizon",
    )

    def apply_overrides(self, config: "TrainConfig", pretrained_step: int | None = None) -> "TrainConfig":
        """Apply all non-None overrides from this FineTuneConfig to the given TrainConfig.

        Handles data overrides (data_dir, dataset_name, assets),
        direct field overrides (save_interval, log_interval, etc.), and — when
        pretrained_step is provided — num_train_steps (offset to absolute) and
        lr_schedule (wrapped with step offset).
        """
        config = self._apply_data_overrides(config)
        if self.model_overrides:
            config = dataclasses.replace(config, model = dataclasses.replace(config.model, **self.model_overrides))
            logging.info("Applied FineTuneConfig model_overrides: %s", list(self.model_overrides.keys()))
        if self.policy_overrides:
            if config.policy is None:
                raise ValueError(
                    "FineTuneConfig.policy_overrides is set but config.policy is None — "
                    "the base TrainConfig must register a policy before it can be overridden.",
                )
            config = dataclasses.replace(config, policy = dataclasses.replace(config.policy, **self.policy_overrides))
            logging.info("Applied FineTuneConfig policy_overrides: %s", list(self.policy_overrides.keys()))

        replacements: dict[str, Any] = {}
        for field in self._TRAIN_CONFIG_FIELDS:
            value = getattr(self, field)
            if value is not None:
                replacements[field] = value

        if pretrained_step is not None:
            if self.num_train_steps is not None:
                replacements["num_train_steps"] = pretrained_step + self.num_train_steps
            if self.lr_schedule is not None:
                replacements["lr_schedule"] = _optimizer.OffsetSchedule(
                    base = self.lr_schedule, offset = pretrained_step,
                )

        if replacements:
            config = dataclasses.replace(config, **replacements)
            logging.info("Applied FineTuneConfig overrides: %s", list(replacements.keys()))

        return config

    def initialize(
        self,
        config: "TrainConfig",
        pretrained_step: int,
        train_state: Any,
        mesh: Any,
    ) -> tuple["TrainConfig", Any, Any, Any, bool]:
        """Full fine-tuning initialization: apply overrides, create optimizer, and checkpoint manager.

        Returns (config, train_state, train_state_sharding, checkpoint_manager, ft_resuming).
        The caller is responsible for restoring from the FT checkpoint when ft_resuming is True,
        because the save structure (plain TrainState vs ActorCriticTrainState) is caller-specific.
        """
        import openpi.training.checkpoints as _checkpoints
        import openpi.training.sharding as _sharding

        config = self.apply_overrides(config, pretrained_step = pretrained_step)

        ft_tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask = None)
        ft_opt_state = ft_tx.init(train_state.params.filter(config.trainable_filter))
        train_state = train_state.replace(tx = ft_tx, opt_state = ft_opt_state)
        train_state_sharding = _sharding.fsdp_sharding(train_state, mesh)

        ft_checkpoint_dir = config.checkpoint_dir / self.name
        checkpoint_manager, ft_resuming = _checkpoints.initialize_checkpoint_dir(
            ft_checkpoint_dir,
            keep_period = config.keep_period,
            overwrite = self.overwrite,
            resume = self.resume,
        )

        logging.info(
            "Fine-tuning: %s steps from pretrained step %s, total steps = %s, checkpoint_dir = %s",
            config.num_train_steps - pretrained_step,
            pretrained_step,
            config.num_train_steps,
            ft_checkpoint_dir,
        )

        return config, train_state, train_state_sharding, checkpoint_manager, ft_resuming

    def _apply_data_overrides(self, config: "TrainConfig") -> "TrainConfig":
        """Apply data_overrides to config.data via dataclasses.replace.

        Special keys handled before passthrough:
        - "data_dir": mapped to rlds_data_dir
        - "dataset_name": split "name:version" into a new datasets tuple
        """
        # Full factory replacement takes precedence over data_overrides.
        if self.data_factory is not None:
            logging.info(
                "Applied FineTuneConfig data_factory replacement: %s -> %s",
                type(config.data).__name__, type(self.data_factory).__name__,
            )
            return dataclasses.replace(config, data = self.data_factory)

        if not self.data_overrides:
            return config

        data_factory = config.data
        if not isinstance(data_factory, RoboCoinRldsDataConfig | Hdf5RldsDataConfig | RLDSRoboCasaDataConfig):
            raise TypeError(
                f"FineTuneConfig data overrides are only supported for RoboCoinRldsDataConfig, "
                f"Hdf5RldsDataConfig, or RLDSRoboCasaDataConfig, got {type(data_factory).__name__}"
            )

        overrides = dict(self.data_overrides)
        replacements: dict[str, Any] = {}

        if "data_dir" in overrides:
            replacements["rlds_data_dir"] = overrides.pop("data_dir")

        if "dataset_name" in overrides:
            dataset_name = overrides.pop("dataset_name")
            parts = dataset_name.split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"dataset_name must be in 'name:version' format, got '{dataset_name}'"
                )
            name, version = parts
            base_dataset = data_factory.datasets[0]
            new_dataset = dataclasses.replace(base_dataset, name = name, version = version)
            replacements["datasets"] = (new_dataset,)

        if "assets" in overrides:
            replacements["assets"] = overrides.pop("assets")

        replacements.update(overrides)

        new_data = dataclasses.replace(data_factory, **replacements)
        return dataclasses.replace(config, data = new_data)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name (defaults to config name).
    exp_name: str | None = None

    # Model or value function config.
    model: _model.BaseModelConfig | _value_functions_base.BaseValueFunctionConfig = dataclasses.field(
        default_factory=pi0_config.Pi0Config
    )

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    policy_lr_schedule: _optimizer.LRScheduleConfig | None = None
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = None

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 86
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 1000
    # How often (in steps) to save checkpoints.
    save_interval: int = 10000
    # How often (in steps) to generate validation plots.
    plot_interval: int = 50000
    # Number of validation trajectories to use for plotting.
    num_val_trajectories: int = 5
    # Repo IDs guaranteed to appear in validation plots. Must have length < num_val_trajectories.
    include_repos: tuple[str, ...] = ()
    # Optional directory to cache validation episodes. If not set, it defaults to {checkpoint_dir}/val_episodes.
    validation_cache_dir: str | None = None
    # Checkpoints matching step % keep_period == 0 will be preserved.
    keep_period: int | None = 100000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True
    # Optional wandb group name for organizing runs within a project.
    wandb_group: str | None = None
    # If true, on --resume start a brand-new wandb run (and overwrite
    # wandb_id.txt) instead of continuing the existing run. Useful when logs
    # have advanced past the checkpoint step and would collide with wandb's
    # monotonic step requirement.
    wandb_new: bool = False

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # Backbone variant for tokenizer selection: "gemma<i>" uses Gemma<i>Tokenizer,
    # None uses PaligemmaTokenizer.
    backbone_variant: str | None = None
    # Number of actions in the action chunk. None means V(s), not Q(s,a).
    action_horizon: int | None = None

    # Maximum gradient norm for clipping. None = no clipping.
    clip_grad_norm: float | None = None

    # === Policy Training (Actor-Critic) ===
    # Optional policy model for actor-critic training. When set, the training script
    # can train both a value function (critic) and a policy (actor).
    policy: _model.BaseModelConfig | None = None

    # Policy extraction objective config. Determines how the policy is trained.
    # Use NoopPolicyConfig for critic-only training, AWRPolicyConfig for IQL-style
    # policy extraction, or DDPGPolicyConfig for DDPG-style policy improvement.
    policy_extraction: _policy_extraction.BasePolicyExtractionConfig | None = None

    # Critic steps per policy step. Controls the ratio of critic to policy updates.
    # - N > 0: Update policy every N critic steps
    # - 0: Frozen critic mode (requires weight_loader to load critic checkpoint)
    critic_steps_per_policy_step: int = 1

    # === Fine-Tuning / Validation-Only Mode ===
    # Name of a FineTuneConfig to apply. When set, overrides dataset, schedule, and intervals.
    fine_tune: str | None = None

    # === Policy Evaluation ===
    # How often (in training steps) to run policy evaluation. 0 = disabled.
    eval_interval: int = 100000
    # Evaluation environment config. If None, no evaluation is performed.
    eval_env: EvalEnvConfig | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> epath.Path:
        """Get the checkpoint directory for this config."""
        exp_name = self.exp_name if self.exp_name else self.name
        base_path = epath.Path(self.checkpoint_base_dir)
        full_path = base_path / self.name / exp_name
        # Only resolve local paths - GCS paths (gs://) should not be resolved
        if "gs://" not in str(full_path):
            full_path = full_path.resolve()
        return full_path

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.fine_tune is not None and self.fine_tune not in _FINE_TUNE_CONFIGS_DICT:
            closest = difflib.get_close_matches(self.fine_tune, _FINE_TUNE_CONFIGS_DICT.keys(), n = 1, cutoff = 0.0)
            closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
            raise ValueError(f"FineTuneConfig '{self.fine_tune}' not found.{closest_str}")


def _make_antmaze_large_diverse_configs() -> list[TrainConfig]:
    """Create antmaze-large-diverse-v1 configs."""
    # Get dimensions from Minari dataset environment spec
    try:
        state_dim, action_dim, action_low, action_high = minari_utils.get_minari_dims("D4RL/antmaze/large-diverse-v1")
    except ImportError:
        logging.warning("minari not installed, skipping antmaze configs")
        return []

    return [
        # MLP BC config
        TrainConfig(
            name="antmaze_large_diverse_v1_mlp_bc",
            model=mlp_config.MLPConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                action_horizon=1,
                hidden_dims=(256, 256),
                action_low=tuple(action_low.tolist()),
                action_high=tuple(action_high.tolist()),
            ),
            data=D4RLDataConfig(
                repo_id="debug/minari_D4RL_antmaze_large_diverse_v1",
                default_task="antmaze-large-diverse-v1",
            ),
            num_train_steps=50_000,
            batch_size=256,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=3e-4,
                decay_steps=50_000,
                decay_lr=1e-5,
            ),
            num_workers=0,
        ),
        # MC Q-function with MSE regression - using MinariDataConfig for fast in-memory loading
        TrainConfig(
            name="antmaze_large_diverse_v1_q_regression",
            num_workers=0,
            model=_value_function.MCValueFunctionConfig(
                network_config=_mlp_network.MLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=True,
                    action_dim=action_dim,
                    action_horizon=1,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                head_config=_heads.RegressionHeadConfig(),
            ),
            data=MinariDataConfig(
                minari_dataset_id="D4RL/antmaze/large-diverse-v1",
                discount=0.99,
                reward_bias=-1.0,
            ),
            num_train_steps=1_000_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # MC Q-function with HL-Gauss categorical loss - using MinariDataConfig for fast in-memory loading
        TrainConfig(
            name="antmaze_large_diverse_v1_q_hl_gauss",
            num_workers=0,
            model=_value_function.MCValueFunctionConfig(
                network_config=_mlp_network.MLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=True,
                    action_dim=action_dim,
                    action_horizon=1,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                head_config=_heads.CategoricalHeadConfig(
                    v_min=-100.0,
                    v_max=0.0,
                    num_bins=128,
                    sigma=0.75,
                ),
            ),
            data=MinariDataConfig(
                minari_dataset_id="D4RL/antmaze/large-diverse-v1",
                discount=0.99,
                reward_bias=-1.0,
            ),
            num_train_steps=1_000_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # MLP SARSA Q-function with MSE regression
        TrainConfig(
            name="antmaze_large_diverse_v1_sarsa_regression",
            num_workers=0,
            model=_value_function.SARSAValueFunctionConfig(
                network_config=_mlp_network.MLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=True,
                    action_dim=action_dim,
                    action_horizon=1,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                head_config=_heads.RegressionHeadConfig(),
                discount=0.99,
            ),
            data=MinariDataConfig(
                minari_dataset_id="D4RL/antmaze/large-diverse-v1",
                discount=0.99,
                reward_bias=-1.0,
            ),
            num_train_steps=1_000_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # MLP SARSA Q-function with HL-Gauss categorical loss
        TrainConfig(
            name="antmaze_large_diverse_v1_sarsa_hl_gauss",
            num_workers=0,
            model=_value_function.SARSAValueFunctionConfig(
                network_config=_mlp_network.MLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=True,
                    action_dim=action_dim,
                    action_horizon=1,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                head_config=_heads.CategoricalHeadConfig(
                    v_min=-100.0,
                    v_max=0.0,
                    num_bins=128,
                    sigma=0.75,
                ),
                discount=0.99,
            ),
            data=MinariDataConfig(
                minari_dataset_id="D4RL/antmaze/large-diverse-v1",
                discount=0.99,
                reward_bias=-1.0,
            ),
            num_train_steps=1_000_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # Multi-Transition MC Q-function (8 transitions per sample)
        TrainConfig(
            name="antmaze_large_diverse_v1_multi_mc",
            num_workers=0,
            model=_value_function.MultiMCValueFunctionConfig(
                network_config=_mlp_network.MultiMLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=True,
                    action_dim=action_dim,
                    action_horizon=1,
                    num_transitions_per_sample=8,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                head_config=_heads.RegressionHeadConfig(),
            ),
            data=MultiTransitionMinariDataConfig(
                minari_dataset_id="D4RL/antmaze/large-diverse-v1",
                discount=0.99,
                reward_bias=-1.0,
                num_transitions_per_sample=8,
                multi_transition_sampler_type="trajectory_uniform",
            ),
            num_train_steps=1_000_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # Multi-Transition MC Q-function with consecutive sampling (8 transitions per sample)
        TrainConfig(
            name="antmaze_large_diverse_v1_multi_mc_consecutive",
            num_workers=0,
            model=_value_function.MultiMCValueFunctionConfig(
                network_config=_mlp_network.MultiMLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=True,
                    action_dim=action_dim,
                    action_horizon=1,
                    num_transitions_per_sample=8,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                head_config=_heads.RegressionHeadConfig(),
            ),
            data=MultiTransitionMinariDataConfig(
                minari_dataset_id="D4RL/antmaze/large-diverse-v1",
                discount=0.99,
                reward_bias=-1.0,
                num_transitions_per_sample=8,
                multi_transition_sampler_type="trajectory_consecutive",
            ),
            num_train_steps=1_000_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # IQL with Q-ensemble of size 2
        TrainConfig(
            name="antmaze_large_diverse_v1_iql",
            num_workers=0,
            model=_value_function.IQLValueFunctionConfig(
                q_network_config=_ensemble_network.EnsembleNetworkConfig(
                    base_config=_mlp_network.MLPNetworkConfig(
                        state_dim=state_dim,
                        action_conditioned=True,
                        action_dim=action_dim,
                        action_horizon=1,
                        hidden_dims=(256, 256, 256, 256),
                        use_layer_norm=False,
                    ),
                    ensemble_size=2,
                ),
                v_network_config=_mlp_network.MLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=False,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                q_head_config=_heads.EnsembleHeadConfig(
                    base_config=_heads.RegressionHeadConfig(),
                    ensemble_size=2,
                ),
                v_head_config=_heads.RegressionHeadConfig(),
                expectile=0.9,
                discount=0.99,
                tau=0.005,
            ),
            data=MinariDataConfig(
                minari_dataset_id="D4RL/antmaze/large-diverse-v1",
                discount=0.99,
                reward_bias=-1.0,
            ),
            num_train_steps=1_000_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # IQL with AWR policy training
        TrainConfig(
            name="antmaze_large_diverse_v1_iql_awr",
            model=_value_function.IQLValueFunctionConfig(
                q_network_config=_ensemble_network.EnsembleNetworkConfig(
                    base_config=_mlp_network.MLPNetworkConfig(
                        state_dim=state_dim,
                        action_conditioned=True,
                        action_dim=action_dim,
                        action_horizon=1,
                        hidden_dims=(256, 256),
                        use_layer_norm=False,
                    ),
                    ensemble_size=2,
                ),
                v_network_config=_mlp_network.MLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=False,
                    hidden_dims=(256, 256),
                    use_layer_norm=False,
                ),
                q_head_config=_heads.EnsembleHeadConfig(
                    base_config=_heads.RegressionHeadConfig(),
                    ensemble_size=2,
                ),
                v_head_config=_heads.RegressionHeadConfig(),
                expectile=0.9,
                discount=0.99,
                tau=0.005,
            ),
            policy=_tanh_gaussian.TanhGaussianConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                action_horizon=1,
                hidden_dims=(256, 256),
                state_dependent_std=False,
                log_std_min=-5.0,
                log_std_max=2.0,
            ),
            policy_extraction=_policy_extraction.AWRPolicyConfig(
                temperature=10.0,
                clip_exp=100.0,
            ),
            data=MinariDataConfig(
                minari_dataset_id="D4RL/antmaze/large-diverse-v1",
                discount=0.99,
                reward_bias=-1.0,
            ),
            num_train_steps=1_000_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
            policy_lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=3e-4, decay_steps=1_000_000, decay_lr=0.0),
            eval_interval=100000,
            eval_env=MinariEvalEnvConfig(num_eval_episodes=16),
            num_workers=0,
        ),
        # Multi-IQL with Q-ensemble (consecutive sampling)
        TrainConfig(
            name="antmaze_large_diverse_v1_multi_iql_consecutive",
            num_workers=0,
            model=_value_function.MultiIQLValueFunctionConfig(
                q_network_config=_ensemble_network.EnsembleMultiNetworkConfig(
                    base_config=_mlp_network.MultiMLPNetworkConfig(
                        state_dim=state_dim,
                        action_conditioned=True,
                        action_dim=action_dim,
                        action_horizon=1,
                        num_transitions_per_sample=8,
                        hidden_dims=(256, 256, 256, 256),
                        use_layer_norm=False,
                    ),
                    ensemble_size=2,
                ),
                v_network_config=_mlp_network.MultiMLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=False,
                    num_transitions_per_sample=8,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                q_head_config=_heads.EnsembleHeadConfig(
                    base_config=_heads.RegressionHeadConfig(),
                    ensemble_size=2,
                ),
                v_head_config=_heads.RegressionHeadConfig(),
                expectile=0.9,
                discount=0.99,
                tau=0.005,
            ),
            data=MultiTransitionMinariDataConfig(
                minari_dataset_id="D4RL/antmaze/large-diverse-v1",
                discount=0.99,
                reward_bias=-1.0,
                num_transitions_per_sample=8,
                multi_transition_sampler_type="trajectory_consecutive",
            ),
            num_train_steps=1_000_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # Multi-IQL with Q-ensemble (random sampling)
        TrainConfig(
            name="antmaze_large_diverse_v1_multi_iql_random",
            num_workers=0,
            model=_value_function.MultiIQLValueFunctionConfig(
                q_network_config=_ensemble_network.EnsembleMultiNetworkConfig(
                    base_config=_mlp_network.MultiMLPNetworkConfig(
                        state_dim=state_dim,
                        action_conditioned=True,
                        action_dim=action_dim,
                        action_horizon=1,
                        num_transitions_per_sample=8,
                        hidden_dims=(256, 256, 256, 256),
                        use_layer_norm=False,
                    ),
                    ensemble_size=2,
                ),
                v_network_config=_mlp_network.MultiMLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=False,
                    num_transitions_per_sample=8,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                q_head_config=_heads.EnsembleHeadConfig(
                    base_config=_heads.RegressionHeadConfig(),
                    ensemble_size=2,
                ),
                v_head_config=_heads.RegressionHeadConfig(),
                expectile=0.9,
                discount=0.99,
                tau=0.005,
            ),
            data=MultiTransitionMinariDataConfig(
                minari_dataset_id="D4RL/antmaze/large-diverse-v1",
                discount=0.99,
                reward_bias=-1.0,
                num_transitions_per_sample=8,
                multi_transition_sampler_type="trajectory_uniform",
            ),
            num_train_steps=1_000_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
    ]


def _make_pointmaze_large_configs() -> list[TrainConfig]:
    """Create pointmaze-large-v2 configs."""
    # Get dimensions from Minari dataset environment spec
    try:
        state_dim, action_dim, action_low, action_high = minari_utils.get_minari_dims("D4RL/pointmaze/large-v2")
    except ImportError:
        logging.warning("minari not installed, skipping pointmaze configs")
        return []

    return [
        # MC Q-function with MSE regression
        TrainConfig(
            name="pointmaze_large_v2_q_regression",
            num_workers=0,
            model=_value_function.MCValueFunctionConfig(
                network_config=_mlp_network.MLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=True,
                    action_dim=action_dim,
                    action_horizon=1,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                head_config=_heads.RegressionHeadConfig(),
            ),
            data=MinariDataConfig(
                minari_dataset_id="D4RL/pointmaze/large-v2",
                discount=0.99,
                reward_bias=-1.0,
            ),
            num_train_steps=100_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # MLP SARSA Q-function with MSE regression
        TrainConfig(
            name="pointmaze_large_v2_q_sarsa",
            num_workers=0,
            model=_value_function.SARSAValueFunctionConfig(
                network_config=_mlp_network.MLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=True,
                    action_dim=action_dim,
                    action_horizon=1,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                head_config=_heads.RegressionHeadConfig(),
                discount=0.99,
            ),
            data=MinariDataConfig(
                minari_dataset_id="D4RL/pointmaze/large-v2",
                discount=0.99,
                reward_bias=-1.0,
            ),
            num_train_steps=100_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # Multi-MC Q-function (8 transitions per sample)
        TrainConfig(
            name="pointmaze_large_v2_q_multi_mc",
            num_workers=0,
            model=_value_function.MultiMCValueFunctionConfig(
                network_config=_mlp_network.MultiMLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=True,
                    action_dim=action_dim,
                    action_horizon=1,
                    num_transitions_per_sample=8,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                head_config=_heads.RegressionHeadConfig(),
            ),
            data=MultiTransitionMinariDataConfig(
                minari_dataset_id="D4RL/pointmaze/large-v2",
                discount=0.99,
                reward_bias=-1.0,
                num_transitions_per_sample=8,
                multi_transition_sampler_type="trajectory_uniform",
            ),
            num_train_steps=100_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # Multi-MC Q-function with consecutive sampling (8 transitions per sample)
        TrainConfig(
            name="pointmaze_large_v2_q_multi_mc_consecutive",
            num_workers=0,
            model=_value_function.MultiMCValueFunctionConfig(
                network_config=_mlp_network.MultiMLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=True,
                    action_dim=action_dim,
                    action_horizon=1,
                    num_transitions_per_sample=8,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                head_config=_heads.RegressionHeadConfig(),
            ),
            data=MultiTransitionMinariDataConfig(
                minari_dataset_id="D4RL/pointmaze/large-v2",
                discount=0.99,
                reward_bias=-1.0,
                num_transitions_per_sample=8,
                multi_transition_sampler_type="trajectory_consecutive",
            ),
            num_train_steps=100_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # Multi-SARSA Q-function with consecutive sampling (regression)
        TrainConfig(
            name="pointmaze_large_v2_q_multi_sarsa_consecutive",
            num_workers=0,
            model=_value_function.MultiSARSAValueFunctionConfig(
                network_config=_mlp_network.MultiMLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=True,
                    action_dim=action_dim,
                    action_horizon=1,
                    num_transitions_per_sample=8,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                head_config=_heads.RegressionHeadConfig(),
                discount=0.99,
            ),
            data=MultiTransitionMinariDataConfig(
                minari_dataset_id="D4RL/pointmaze/large-v2",
                discount=0.99,
                reward_bias=-1.0,
                num_transitions_per_sample=8,
                multi_transition_sampler_type="trajectory_consecutive",
            ),
            num_train_steps=100_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # Multi-SARSA Q-function with consecutive sampling (HL-Gauss)
        TrainConfig(
            name="pointmaze_large_v2_q_multi_sarsa_hl_gauss_consecutive",
            num_workers=0,
            model=_value_function.MultiSARSAValueFunctionConfig(
                network_config=_mlp_network.MultiMLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=True,
                    action_dim=action_dim,
                    action_horizon=1,
                    num_transitions_per_sample=8,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                head_config=_heads.CategoricalHeadConfig(
                    v_min=-100.0,
                    v_max=0.0,
                    num_bins=128,
                    sigma=0.75,
                ),
                discount=0.99,
            ),
            data=MultiTransitionMinariDataConfig(
                minari_dataset_id="D4RL/pointmaze/large-v2",
                discount=0.99,
                reward_bias=-1.0,
                num_transitions_per_sample=8,
                multi_transition_sampler_type="trajectory_consecutive",
            ),
            num_train_steps=100_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # IQL with Q-ensemble of size 2
        TrainConfig(
            name="pointmaze_large_v2_iql",
            num_workers=0,
            model=_value_function.IQLValueFunctionConfig(
                q_network_config=_ensemble_network.EnsembleNetworkConfig(
                    base_config=_mlp_network.MLPNetworkConfig(
                        state_dim=state_dim,
                        action_conditioned=True,
                        action_dim=action_dim,
                        action_horizon=1,
                        hidden_dims=(256, 256, 256, 256),
                        use_layer_norm=False,
                    ),
                    ensemble_size=2,
                ),
                v_network_config=_mlp_network.MLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=False,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                q_head_config=_heads.EnsembleHeadConfig(
                    base_config=_heads.RegressionHeadConfig(),
                    ensemble_size=2,
                ),
                v_head_config=_heads.RegressionHeadConfig(),
                expectile=0.9,
                discount=0.99,
                tau=0.005,
            ),
            policy=_tanh_gaussian.TanhGaussianConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                action_horizon=1,
                hidden_dims=(256, 256),
                state_dependent_std=False,
                log_std_min=-5.0,
                log_std_max=2.0,
            ),
            policy_extraction=_policy_extraction.AWRPolicyConfig(
                temperature=10.0,
                clip_exp=100.0,
            ),
            data=MinariDataConfig(
                minari_dataset_id="D4RL/pointmaze/large-v2",
                discount=0.99,
                reward_bias=-1.0,
            ),
            num_train_steps=100_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
            policy_lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=3e-4, decay_steps=100_000, decay_lr=0.0),
            eval_interval=0,
            eval_env=MinariEvalEnvConfig(num_eval_episodes=16, record_video=False),
        ),
        # Multi-IQL with Q-ensemble (consecutive sampling)
        TrainConfig(
            name="pointmaze_large_v2_multi_iql_consecutive",
            num_workers=0,
            model=_value_function.MultiIQLValueFunctionConfig(
                q_network_config=_ensemble_network.EnsembleMultiNetworkConfig(
                    base_config=_mlp_network.MultiMLPNetworkConfig(
                        state_dim=state_dim,
                        action_conditioned=True,
                        action_dim=action_dim,
                        action_horizon=1,
                        num_transitions_per_sample=8,
                        hidden_dims=(256, 256, 256, 256),
                        use_layer_norm=False,
                    ),
                    ensemble_size=2,
                ),
                v_network_config=_mlp_network.MultiMLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=False,
                    num_transitions_per_sample=8,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                q_head_config=_heads.EnsembleHeadConfig(
                    base_config=_heads.RegressionHeadConfig(),
                    ensemble_size=2,
                ),
                v_head_config=_heads.RegressionHeadConfig(),
                expectile=0.9,
                discount=0.99,
                tau=0.005,
            ),
            data=MultiTransitionMinariDataConfig(
                minari_dataset_id="D4RL/pointmaze/large-v2",
                discount=0.99,
                reward_bias=-1.0,
                num_transitions_per_sample=8,
                multi_transition_sampler_type="trajectory_consecutive",
            ),
            num_train_steps=100_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
        # Multi-IQL with Q-ensemble (random sampling)
        TrainConfig(
            name="pointmaze_large_v2_multi_iql_random",
            num_workers=0,
            model=_value_function.MultiIQLValueFunctionConfig(
                q_network_config=_ensemble_network.EnsembleMultiNetworkConfig(
                    base_config=_mlp_network.MultiMLPNetworkConfig(
                        state_dim=state_dim,
                        action_conditioned=True,
                        action_dim=action_dim,
                        action_horizon=1,
                        num_transitions_per_sample=8,
                        hidden_dims=(256, 256, 256, 256),
                        use_layer_norm=False,
                    ),
                    ensemble_size=2,
                ),
                v_network_config=_mlp_network.MultiMLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=False,
                    num_transitions_per_sample=8,
                    hidden_dims=(256, 256, 256, 256),
                    use_layer_norm=False,
                ),
                q_head_config=_heads.EnsembleHeadConfig(
                    base_config=_heads.RegressionHeadConfig(),
                    ensemble_size=2,
                ),
                v_head_config=_heads.RegressionHeadConfig(),
                expectile=0.9,
                discount=0.99,
                tau=0.005,
            ),
            data=MultiTransitionMinariDataConfig(
                minari_dataset_id="D4RL/pointmaze/large-v2",
                discount=0.99,
                reward_bias=-1.0,
                num_transitions_per_sample=8,
                multi_transition_sampler_type="trajectory_uniform",
            ),
            num_train_steps=100_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
        ),
    ]


def _make_antmaze_large_diverse_v2_legacy_configs() -> list[TrainConfig]:
    """Create antmaze-large-diverse-v2 configs using legacy D4RL dataset."""
    # Get dimensions from legacy D4RL environment
    if legacy_d4rl_utils is None:
        logging.warning("legacy_d4rl_utils not available, skipping legacy D4RL configs")
        return []
    state_dim, action_dim, _, _ = legacy_d4rl_utils.get_legacy_d4rl_dims("antmaze-large-diverse-v2")

    return [
        # IQL with AWR policy training (legacy D4RL)
        TrainConfig(
            name="antmaze_large_diverse_v2_iql_awr",
            model=_value_function.IQLValueFunctionConfig(
                q_network_config=_ensemble_network.EnsembleNetworkConfig(
                    base_config=_mlp_network.MLPNetworkConfig(
                        state_dim=state_dim,
                        action_conditioned=True,
                        action_dim=action_dim,
                        action_horizon=1,
                        hidden_dims=(256, 256),
                        use_layer_norm=False,
                    ),
                    ensemble_size=2,
                ),
                v_network_config=_mlp_network.MLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=False,
                    hidden_dims=(256, 256),
                    use_layer_norm=False,
                ),
                q_head_config=_heads.EnsembleHeadConfig(
                    base_config=_heads.RegressionHeadConfig(),
                    ensemble_size=2,
                ),
                v_head_config=_heads.RegressionHeadConfig(),
                expectile=0.9,
                discount=0.99,
                tau=0.005,
            ),
            policy=_tanh_gaussian.TanhGaussianConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                action_horizon=1,
                hidden_dims=(256, 256),
                state_dependent_std=False,
                log_std_min=-5.0,
                log_std_max=2.0,
            ),
            policy_extraction=_policy_extraction.AWRPolicyConfig(
                temperature=10.0,
                clip_exp=100.0,
            ),
            data=LegacyD4RLDataConfig(
                legacy_d4rl_env_name="antmaze-large-diverse-v2",
                discount=0.99,
                reward_bias=-1.0,
            ),
            num_train_steps=1_000_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
            policy_lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=3e-4, decay_steps=1_000_000, decay_lr=0.0),
            eval_interval=100000,
            eval_env=LegacyD4RLEvalEnvConfig(num_eval_episodes=16),
            num_workers=0,
        ),
        # Multi-IQL with AWR policy training (legacy D4RL, consecutive sampling)
        TrainConfig(
            name="antmaze_large_diverse_v2_iql_awr_multi_consecutive",
            model=_value_function.MultiIQLValueFunctionConfig(
                q_network_config=_ensemble_network.EnsembleMultiNetworkConfig(
                    base_config=_mlp_network.MultiMLPNetworkConfig(
                        state_dim=state_dim,
                        action_conditioned=True,
                        action_dim=action_dim,
                        action_horizon=1,
                        num_transitions_per_sample=8,
                        hidden_dims=(256, 256),
                        use_layer_norm=False,
                    ),
                    ensemble_size=2,
                ),
                v_network_config=_mlp_network.MultiMLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=False,
                    num_transitions_per_sample=8,
                    hidden_dims=(256, 256),
                    use_layer_norm=False,
                ),
                q_head_config=_heads.EnsembleHeadConfig(
                    base_config=_heads.RegressionHeadConfig(),
                    ensemble_size=2,
                ),
                v_head_config=_heads.RegressionHeadConfig(),
                expectile=0.9,
                discount=0.99,
                tau=0.005,
            ),
            policy=_tanh_gaussian.TanhGaussianConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                action_horizon=1,
                hidden_dims=(256, 256),
                state_dependent_std=False,
                log_std_min=-5.0,
                log_std_max=2.0,
            ),
            policy_extraction=_policy_extraction.MultiAWRPolicyConfig(
                temperature=10.0,
                clip_exp=100.0,
            ),
            data=MultiTransitionLegacyD4RLDataConfig(
                legacy_d4rl_env_name="antmaze-large-diverse-v2",
                discount=0.99,
                reward_bias=-1.0,
                num_transitions_per_sample=8,
                multi_transition_sampler_type="trajectory_consecutive",
            ),
            num_train_steps=1_000_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
            policy_lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=3e-4, decay_steps=1_000_000, decay_lr=0.0),
            eval_interval=100000,
            eval_env=LegacyD4RLEvalEnvConfig(num_eval_episodes=16),
            num_workers=0,
        ),
        # Multi-IQL with AWR policy training (legacy D4RL, random sampling)
        TrainConfig(
            name="antmaze_large_diverse_v2_iql_awr_multi_random",
            model=_value_function.MultiIQLValueFunctionConfig(
                q_network_config=_ensemble_network.EnsembleMultiNetworkConfig(
                    base_config=_mlp_network.MultiMLPNetworkConfig(
                        state_dim=state_dim,
                        action_conditioned=True,
                        action_dim=action_dim,
                        action_horizon=1,
                        num_transitions_per_sample=8,
                        hidden_dims=(256, 256),
                        use_layer_norm=False,
                    ),
                    ensemble_size=2,
                ),
                v_network_config=_mlp_network.MultiMLPNetworkConfig(
                    state_dim=state_dim,
                    action_conditioned=False,
                    num_transitions_per_sample=8,
                    hidden_dims=(256, 256),
                    use_layer_norm=False,
                ),
                q_head_config=_heads.EnsembleHeadConfig(
                    base_config=_heads.RegressionHeadConfig(),
                    ensemble_size=2,
                ),
                v_head_config=_heads.RegressionHeadConfig(),
                expectile=0.9,
                discount=0.99,
                tau=0.005,
            ),
            policy=_tanh_gaussian.TanhGaussianConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                action_horizon=1,
                hidden_dims=(256, 256),
                state_dependent_std=False,
                log_std_min=-5.0,
                log_std_max=2.0,
            ),
            policy_extraction=_policy_extraction.MultiAWRPolicyConfig(
                temperature=10.0,
                clip_exp=100.0,
            ),
            data=MultiTransitionLegacyD4RLDataConfig(
                legacy_d4rl_env_name="antmaze-large-diverse-v2",
                discount=0.99,
                reward_bias=-1.0,
                num_transitions_per_sample=8,
                multi_transition_sampler_type="trajectory_uniform",
            ),
            num_train_steps=1_000_000,
            batch_size=256,
            lr_schedule=_optimizer.ConstantSchedule(lr=3e-4),
            policy_lr_schedule=_optimizer.CosineDecaySchedule(peak_lr=3e-4, decay_steps=1_000_000, decay_lr=0.0),
            eval_interval=100000,
            eval_env=LegacyD4RLEvalEnvConfig(num_eval_episodes=16),
            num_workers=0,
        ),
    ]


def _make_antmaze_large_diverse_v2_legacy_configs_safe() -> list[TrainConfig]:
    try:
        return _make_antmaze_large_diverse_v2_legacy_configs()
    except Exception:
        logging.warning("Failed to initialize legacy Antmaze configs (likely missing mujoco_py). Skipping.")
        return []


# =============================================================================
# Fine-tune configs
# =============================================================================

_FINE_TUNE_CONFIGS: list[FineTuneConfig] = [
    # Fine-tune the RoboCOIN-pretrained Q-SARSA value function on RoboCasa.
    # Use the no-suffix `robocasa_paligemma_q_sarsa_finetune` variant below for TPU runs
    # (GCS paths). The base config provides the model architecture and weights are loaded
    # from the base TrainConfig's checkpoint dir; this FineTuneConfig only swaps the data
    # factory to RLDSRoboCasaDataConfig (single-arm → bimanual right-arm slot,
    # base/control_mode dropped, right_wrist masked) and lowers the LR.
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_finetune_gpu",
        data_factory = RLDSRoboCasaDataConfig(
            rlds_data_dir = "/data/group_data/rl/datasets/robocasa_rlds",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__close_blender_lid", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "/data/group_data/rl/saksham3/datasets/robocasa/norm_stats",
                asset_id = "target__atomic__close_blender_lid",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            image_size = 224,
            discount = 0.999,
            # FPS interpolation: 20 Hz native → 30 Hz target (matches RoboCOIN-pretrained
            # value head's training rate). action_horizon_seconds=1.0 → 30 valid frames.
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            replace_boundary_actions = False,
            shuffle_buffer_size = 100_000,
        ),
        num_train_steps = 5_000,
        save_interval = 2_500,
        plot_interval = 2_500,
        keep_period = 2_500,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/data/user_data/saksham3/generalist_value_function/validation_cache_dir_robocasa/",
        include_repos = (),
    ),
    # GCS-paths mirror of robocasa_paligemma_q_sarsa_finetune_gpu for TPU runs.
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_finetune_close_blender_lid",
        data_factory = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__close_blender_lid", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__close_blender_lid",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            image_size = 224,
            discount = 0.999,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            replace_boundary_actions = False,
            shuffle_buffer_size = 50_000,
            use_quantile_norm = False,
        ),
        num_train_steps = 5_000,
        save_interval = 2_500,
        plot_interval = 2_500,
        keep_period = 2_500,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir/",
        include_repos = (),
    ),
    # Composite-task fine-tune variant. target__composite__arrange_tea decomposes into
    # 3 subtasks; the subtask tracker (robocasa_subtask_tracker.py) recovers the per-step
    # subtask from the raw state, so steps_to_subtask_end counts down to the current
    # subtask end and the prompt is the current subtask description.
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_finetune_arrange_tea",
        data_factory = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__composite__arrange_tea", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__composite__arrange_tea",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            image_size = 224,
            discount = 0.999,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            replace_boundary_actions = False,
            shuffle_buffer_size = 50_000,
            use_quantile_norm = False,
        ),
        num_train_steps = 5_000,
        save_interval = 2_500,
        plot_interval = 2_500,
        keep_period = 2_500,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_arrange_tea/",
        include_repos = (),
    ),
    # Per-task non-chunk-wise-delta fine-tune variants for fine-tuning the
    # robocoin_bimanual_paligemma_q_sarsa base. Mirrors the base's design:
    # absolute actions (no chunk-wise delta), z-score normalization, no_state
    # backbone. Both mask_boundary_actions and replace_boundary_actions are
    # False, so the action_mask is the 30/20 fps-tail mask with no boundary
    # truncation/replacement.
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_finetune_coffee_setup_mug",
        data_factory = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__coffee_setup_mug", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__coffee_setup_mug",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            image_size = 224,
            discount = 0.999,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = False,
            use_quantile_norm = False,
            shuffle_buffer_size = 50_000,
        ),
        num_train_steps = 5_000,
        save_interval = 2_500,
        plot_interval = 2_500,
        keep_period = 2_500,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_coffee_setup_mug/",
        include_repos = (),
    ),
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_finetune_turn_on_sink_faucet",
        data_factory = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__turn_on_sink_faucet", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__turn_on_sink_faucet",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            image_size = 224,
            discount = 0.999,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = False,
            use_quantile_norm = False,
            shuffle_buffer_size = 50_000,
        ),
        num_train_steps = 5_000,
        save_interval = 2_500,
        plot_interval = 2_500,
        keep_period = 2_500,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_turn_on_sink_faucet/",
        include_repos = (),
    ),
    # chunk-wise-delta + quantile-norm variant for fine-tuning the
    # robocoin_bimanual_paligemma_q_sarsa_chunk_wise_delta base.
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_finetune_chunk_wise_delta",
        data_factory = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__close_blender_lid", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__close_blender_lid",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            image_size = 224,
            discount = 0.999,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
        ),
        num_train_steps = 5_000,
        save_interval = 2_500,
        plot_interval = 2_500,
        keep_period = 2_500,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_chunk_wise_delta/",
        include_repos = (),
    ),
    # routes all three RoboCasa cameras into the bimanual image slots instead of masking the
    # third (left/top → base_0_rgb, right/top → left_wrist_0_rgb, wrist_camera → right_wrist_0_rgb).
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_finetune_chunk_wise_delta_use_all_cameras",
        data_factory = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__close_blender_lid", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__close_blender_lid",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            use_all_cameras = True,
            image_size = 224,
            discount = 0.999,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
        ),
        num_train_steps = 5_000,
        save_interval = 2_500,
        plot_interval = 2_500,
        keep_period = 2_500,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_chunk_wise_delta_use_all_cameras/",
        include_repos = (),
    ),
    # Per-task variants of robocasa_paligemma_q_sarsa_finetune_chunk_wise_delta. Only
    # the dataset name, asset_id, and validation_cache_dir differ from the parent.
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_finetune_chunk_wise_delta_coffee_setup_mug",
        data_factory = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__coffee_setup_mug", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__coffee_setup_mug",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            image_size = 224,
            discount = 0.999,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
        ),
        num_train_steps = 5_000,
        save_interval = 2_500,
        plot_interval = 2_500,
        keep_period = 2_500,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_chunk_wise_delta_coffee_setup_mug/",
        include_repos = (),
    ),
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_finetune_chunk_wise_delta_open_cabinet",
        data_factory = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__open_cabinet", version = "1.0.0", weight = 1.0),
            ),  
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__open_cabinet",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            image_size = 224,
            discount = 0.999,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
        ),
        num_train_steps = 5_000,
        save_interval = 2_500,
        plot_interval = 2_500,
        keep_period = 2_500,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_chunk_wise_delta_open_cabinet/",
        include_repos = (),
    ),
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_finetune_chunk_wise_delta_close_fridge",
        data_factory = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__close_fridge", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__close_fridge",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            image_size = 224,
            discount = 0.999,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
        ),
        num_train_steps = 5_000,
        save_interval = 2_500,
        plot_interval = 2_500,
        keep_period = 2_500,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_chunk_wise_delta_close_fridge/",
        include_repos = (),
    ),
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_finetune_chunk_wise_delta_pick_place_sink_to_counter",
        data_factory = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__pick_place_sink_to_counter", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__pick_place_sink_to_counter",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            image_size = 224,
            discount = 0.999,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
        ),
        num_train_steps = 5_000,
        save_interval = 2_500,
        plot_interval = 2_500,
        keep_period = 2_500,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_chunk_wise_delta_pick_place_sink_to_counter/",
        include_repos = (),
    ),
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_finetune_chunk_wise_delta_pick_place_toaster_to_counter",
        data_factory = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__pick_place_toaster_to_counter", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__pick_place_toaster_to_counter",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            image_size = 224,
            discount = 0.999,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
        ),
        num_train_steps = 5_000,
        save_interval = 2_500,
        plot_interval = 2_500,
        keep_period = 2_500,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_chunk_wise_delta_pick_place_toaster_to_counter/",
        include_repos = (),
    ),
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_finetune_chunk_wise_delta_turn_on_sink_faucet",
        data_factory = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__turn_on_sink_faucet", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__turn_on_sink_faucet",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            image_size = 224,
            discount = 0.999,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
        ),
        num_train_steps = 5_000,
        save_interval = 2_500,
        plot_interval = 2_500,
        keep_period = 2_500,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_chunk_wise_delta_turn_on_sink_faucet/",
        include_repos = (),
    ),
    # chunk-wise-delta + quantile-norm Q-SARSA critic fine-tune for the sim_bimanual_assembly task. Mirrors the data-pipeline
    # knobs of robocoin_bimanual_paligemma_q_sarsa_chunk_wise_delta so the
    # restored critic sees the same action/state representation it was
    # pre-trained under.
    FineTuneConfig(
        name = "sim_bimanual_assembly_q_sarsa_finetune_chunk_wise_delta",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "sim_bimanual_assembly",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "sim_bimanual_assembly", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/sim_bimanual_assembly",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
        ),
        model_overrides = {"action_horizon": 60},
        action_horizon = 60,
        num_train_steps = 10_000,
        save_interval = 5_000,
        plot_interval = 5_000,
        keep_period = 5_000,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/sim_bimanual_assembly/validation_cache_dir_sim_bimanual_assembly/",
        include_repos = (),
    ),
    FineTuneConfig(
        name = "real_shirt_hang_q_sarsa_finetune_chunk_wise_delta",
        data_factory = Hdf5RldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
        ),
        model_overrides = {"action_horizon": 60},
        action_horizon = 60,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocoin/validation_cache_dir_real_shirt_hang/",
        num_val_trajectories = 2,
        include_repos = (),
        num_train_steps = 8000,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        save_interval = 2000,
        plot_interval = 2000,
        keep_period = 2000,
    ),
    # task_description_predict_current_subtask fine-tunes of the
    # robocoin_bimanual_paligemma_q_sarsa_task_description critic. Same data-pipeline
    # knobs as the chunk_wise_delta finetunes above, plus prompt_mode set to match the
    # pre-trained critic's subtask_prompt_mode so the tokenizer routes through
    # TokenizeRoboCoinSubtaskPrompt.
    FineTuneConfig(
        name = "sim_bimanual_assembly_q_sarsa_finetune_task_description",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "sim_bimanual_assembly",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "sim_bimanual_assembly", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/sim_bimanual_assembly",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # network_config is rebuilt to override no_state=True (ablation: drop the
        # proprioceptive state token during fine-tune); the rest mirrors the base
        # `robocoin_bimanual_paligemma_q_sarsa_task_description` network config.
        model_overrides = {
            "action_horizon": 60,
            "network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 48,
                action_dim = 14,
                dtype = "float32",
                use_layernorm = True,
                no_state = True,
            ),
        },
        action_horizon = 60,
        num_train_steps = 10_000,
        save_interval = 5_000,
        plot_interval = 5_000,
        keep_period = 5_000,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/sim_bimanual_assembly/validation_cache_dir_sim_bimanual_assembly_task_description/",
        include_repos = (),
    ),
    # Identical to sim_bimanual_assembly_q_sarsa_finetune_task_description but keeps
    # the proprioceptive state token (no_state=False).
    FineTuneConfig(
        name = "sim_bimanual_assembly_q_sarsa_finetune_task_description_state",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "sim_bimanual_assembly",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "sim_bimanual_assembly", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/sim_bimanual_assembly",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # network_config is rebuilt to keep no_state=False (retain the
        # proprioceptive state token); the rest mirrors the base
        # `robocoin_bimanual_paligemma_q_sarsa_task_description` network config.
        model_overrides = {
            "action_horizon": 60,
            "network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 48,
                action_dim = 14,
                dtype = "float32",
                use_layernorm = True,
                no_state = False,
            ),
        },
        action_horizon = 60,
        num_train_steps = 10_000,
        save_interval = 5_000,
        plot_interval = 5_000,
        keep_period = 5_000,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/sim_bimanual_assembly/validation_cache_dir_sim_bimanual_assembly_task_description_state/",
        include_repos = (),
    ),
    # Same as sim_bimanual_assembly_q_sarsa_finetune_task_description_state but
    # trained 5x longer at 5x the LR (50k steps, lr=5e-6) with 3 val trajectories.
    FineTuneConfig(
        name = "sim_bimanual_assembly_q_sarsa_finetune_task_description_long",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "sim_bimanual_assembly",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "sim_bimanual_assembly", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/sim_bimanual_assembly",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        model_overrides = {
            "action_horizon": 60,
            "network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 48,
                action_dim = 14,
                dtype = "float32",
                use_layernorm = True,
                no_state = False,
            ),
        },
        action_horizon = 60,
        num_train_steps = 50_000,
        save_interval = 5_000,
        plot_interval = 5_000,
        keep_period = 5_000,
        lr_schedule = _optimizer.ConstantSchedule(lr = 5e-6),
        num_val_trajectories = 3,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/sim_bimanual_assembly/validation_cache_dir_sim_bimanual_assembly_task_description_long/",
        include_repos = (),
    ),
    # Same as sim_bimanual_assembly_q_sarsa_finetune_task_description_state but configured
    # to fine-tune the gemma4 base (robocoin_bimanual_gemma4_q_sarsa_task_description):
    # 480x480 images via the gemma4_e2b backbone and the lower-shuffle / lower-parallelism
    # data pipeline gemma4 needs for its memory budget.
    FineTuneConfig(
        name = "sim_bimanual_assembly_gemma4_q_sarsa_finetune_task_description_state",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "sim_bimanual_assembly",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "sim_bimanual_assembly", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/sim_bimanual_assembly",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            image_size = (480, 480),
            shuffle_buffer_size = 7_500,
            num_parallel_reads = 4,
            num_parallel_calls = 4,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # network_config mirrors the gemma4 base's network_config (paligemma_variant
        # "gemma4_e2b", 480x480 images) with no_state=False to keep the proprioceptive
        # state token.
        model_overrides = {
            "action_horizon": 60,
            "network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (480, 480),
                max_token_len = 48,
                action_dim = 14,
                dtype = "float32",
                paligemma_variant = "gemma4_e2b",
                use_layernorm = True,
                no_state = False,
            ),
        },
        action_horizon = 60,
        num_train_steps = 10_000,
        save_interval = 5_000,
        plot_interval = 5_000,
        keep_period = 5_000,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 2,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/sim_bimanual_assembly/validation_cache_dir_sim_bimanual_assembly_gemma4_task_description_state/",
        include_repos = (),
    ),
    # Fine-tune the paligemma CQL+Best-of-N base (robocoin_bimanual_paligemma_cql_rlds)
    # on sim_bimanual_assembly.
    FineTuneConfig(
        name = "sim_bimanual_assembly_paligemma_cql_rlds_finetune_task_description",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "sim_bimanual_assembly",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "sim_bimanual_assembly", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/sim_bimanual_assembly",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/sim_bimanual_assembly_pi05/",
            max_token_len = 96,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # q_network_config is byte-identical to the base's
        # q_network_config (224x224 images, no_state=True, default paligemma
        # backbone, no layernorm) — restated so the pretrained checkpoint loads
        # without any shape mismatch.
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
            ),
        },
        # Bump the Best-of-N policy's action_horizon to match the 60-frame
        # data chunks. Without this the policy stays at the pretrain 50 and
        # value_function_objectives.py reshape (~lines 339, 507, 521) blows up
        # on the 60-frame CF candidates.
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 5_000,
        plot_interval = 5_000,
        keep_period = 5_000,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 3,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/sim_bimanual_assembly/validation_cache_dir_sim_bimanual_assembly_paligemma_cql_rlds_finetune/",
        include_repos = (),
    ),
    FineTuneConfig(
        name = "sim_bimanual_assembly_paligemma_cql_rlds_finetune_task_description_final",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "sim_bimanual_assembly",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "sim_bimanual_assembly", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/sim_bimanual_assembly",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/sim_bimanual_assembly_pi05/",
            max_token_len = 96,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # q_network_config is byte-identical to the base's
        # q_network_config (224x224 images, no_state=True, default paligemma
        # backbone, no layernorm) — restated so the pretrained checkpoint loads
        # without any shape mismatch.
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
            ),
        },
        # Bump the Best-of-N policy's action_horizon to match the 60-frame
        # data chunks. Without this the policy stays at the pretrain 50 and
        # value_function_objectives.py reshape (~lines 339, 507, 521) blows up
        # on the 60-frame CF candidates.
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 50_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        lr_schedule = _optimizer.ConstantSchedule(lr = 5e-6),
        num_val_trajectories = 3,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/sim_bimanual_assembly/validation_cache_dir_sim_bimanual_assembly_paligemma_cql_rlds_finetune_final/",
        include_repos = (),
    ),
    # real_shirt_hang twin of the sim_bimanual version:
    # same paligemma CQL+Best-of-N FT recipe, swapped onto the real_shirt_hang HDF5 dataset
    # and its CF cache / norm-stats / validation cache.
    FineTuneConfig(
        name = "real_shirt_hang_paligemma_cql_rlds_finetune_task_description",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "real_shirt_hang",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/real_shirt_hang_pi05/",
            max_token_len = 96,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # q_network_config is byte-identical to the base's
        # q_network_config (224x224 images, no_state=True, default paligemma
        # backbone, no layernorm) — restated so the pretrained checkpoint loads
        # without any shape mismatch.
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
            ),
        },
        # Bump the Best-of-N policy's action_horizon to match the 60-frame
        # data chunks. Without this the policy stays at the pretrain 50 and
        # value_function_objectives.py reshape (~lines 339, 507, 521) blows up
        # on the 60-frame CF candidates.
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 5_000,
        plot_interval = 5_000,
        keep_period = 5_000,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 3,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocoin/validation_cache_dir_real_shirt_hang_paligemma_cql_rlds_finetune/",
        include_repos = (),
    ),
    # real_shirt_hang twin of the sim_bimanual _final config:
    # same recipe as real_shirt_hang_paligemma_cql_rlds_finetune_task_description but with the
    # longer 50k-step schedule and 5e-6 lr from sim_bimanual_assembly_..._final.
    FineTuneConfig(
        name = "real_shirt_hang_paligemma_cql_rlds_finetune_task_description_final",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "real_shirt_hang",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/real_shirt_hang_pi05/",
            max_token_len = 96,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # q_network_config is byte-identical to the base's
        # q_network_config (224x224 images, no_state=True, default paligemma
        # backbone, no layernorm) — restated so the pretrained checkpoint loads
        # without any shape mismatch.
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
            ),
        },
        # Bump the Best-of-N policy's action_horizon to match the 60-frame
        # data chunks. Without this the policy stays at the pretrain 50 and
        # value_function_objectives.py reshape (~lines 339, 507, 521) blows up
        # on the 60-frame CF candidates.
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        # Cosine decay 5e-6 -> 5e-7 over the 50k FT steps. FineTuneConfig wraps
        # this in an OffsetSchedule with offset=pretrained_step, so step 0 of the
        # cosine corresponds to the FT-start absolute step.
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 50_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/real_shirt_hang/validation_cache_dir_real_shirt_hang_paligemma_cql_rlds_finetune_final/",
        include_repos = (),
    ),
    # Same as real_shirt_hang_paligemma_cql_rlds_finetune_task_description_final but
    # with prompt_mode="subtask" (the single subtask fed directly as the text prompt),
    # for fine-tuning the robocoin_bimanual_paligemma_cql_rlds_subtask_no_ntp base.
    FineTuneConfig(
        name = "real_shirt_hang_paligemma_cql_rlds_finetune_subtask_final",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "real_shirt_hang",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/real_shirt_hang_pi05/",
            max_token_len = 96,
            prompt_mode = "subtask",
        ),
        # q_network_config is byte-identical to the base's
        # q_network_config (224x224 images, no_state=True, default paligemma
        # backbone, no layernorm) — restated so the pretrained checkpoint loads
        # without any shape mismatch.
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
            ),
        },
        # Bump the Best-of-N policy's action_horizon to match the 60-frame
        # data chunks. Without this the policy stays at the pretrain 50 and
        # value_function_objectives.py reshape (~lines 339, 507, 521) blows up
        # on the 60-frame CF candidates.
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        # Cosine decay 5e-6 -> 5e-7 over the 50k FT steps. FineTuneConfig wraps
        # this in an OffsetSchedule with offset=pretrained_step, so step 0 of the
        # cosine corresponds to the FT-start absolute step.
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 50_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/real_shirt_hang/validation_cache_dir_real_shirt_hang_paligemma_cql_rlds_finetune_subtask_final/",
        include_repos = (),
    ),
    # Same as real_shirt_hang_paligemma_cql_rlds_finetune_task_description_final but
    # with predict_subtask_ar=True in the restated q_network_config, for fine-tuning the
    # robocoin_bimanual_paligemma_cql_rlds_subtask_ar base. Prompt mode stays
    # "task_description_predict_current_subtask" (same as that base); only the AR flag
    # differs from the plain _task_description_final config.
    FineTuneConfig(
        name = "real_shirt_hang_paligemma_cql_rlds_finetune_subtask_ar_final",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "real_shirt_hang",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/real_shirt_hang_pi05/",
            max_token_len = 96,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # q_network_config is byte-identical to the subtask_ar base's
        # q_network_config (224x224 images, no_state=True, predict_subtask_ar=True,
        # default paligemma backbone, no layernorm) — restated so the pretrained
        # checkpoint loads without any shape mismatch and the AR behavior is preserved.
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
        },
        # Bump the Best-of-N policy's action_horizon to match the 60-frame
        # data chunks. Without this the policy stays at the pretrain 50 and
        # value_function_objectives.py reshape (~lines 339, 507, 521) blows up
        # on the 60-frame CF candidates.
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        # Cosine decay 5e-6 -> 5e-7 over the 20k FT steps. FineTuneConfig wraps
        # this in an OffsetSchedule with offset=pretrained_step, so step 0 of the
        # cosine corresponds to the FT-start absolute step.
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 20_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/real_shirt_hang/validation_cache_dir_real_shirt_hang_paligemma_cql_rlds_finetune_subtask_ar_final/",
        include_repos = (),
    ),
    # LeRobot realworld_xarm_packing twin of the real_shirt_hang subtask_ar CQL FT:
    # same recipe, LeRobotRldsDataConfig (critic_mode, subsample) on the packing
    # dataset; max_token_len=160 (128 for the task + 32 for the predicted subtask).
    FineTuneConfig(
        name = "realworld_xarm_packing_paligemma_cql_rlds_finetune_subtask_ar_final",
        data_factory = LeRobotRldsDataConfig(
            repo_id = "realworld_xarm_packing",
            rlds_data_dir = "gs://saksham-euw4/datasets/realworld_xarm_packing",
            datasets = (
                rlds_dataset.RLDSDataset(name = "realworld_xarm_packing", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/datasets/realworld_xarm_packing",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/realworld_xarm_packing_pi05/",
            max_token_len = 160,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # Restated q_network_config (matching the subtask_ar base) so the pretrained
        # checkpoint loads cleanly; max_token_len bumped to 160 for the longer
        # (task, subtask) concatenation of the packing prompts.
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 160,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
        },
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 20_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/realworld_xarm_packing/validation_cache_dir_realworld_xarm_packing_paligemma_cql_rlds_finetune_subtask_ar_final/",
        include_repos = (),
    ),
    # sim_bimanual_assembly twin of real_shirt_hang_paligemma_cql_rlds_finetune_subtask_ar_final:
    # same subtask_ar FT recipe (predict_subtask_ar=True in the restated q_network_config,
    # prompt_mode "task_description_predict_current_subtask"), swapped onto the
    # sim_bimanual_assembly HDF5 dataset and its CF cache / norm-stats / validation cache.
    # Fine-tunes the robocoin_bimanual_paligemma_cql_rlds_subtask_ar base.
    FineTuneConfig(
        name = "sim_bimanual_assembly_paligemma_cql_rlds_finetune_subtask_ar_final",
        data_factory = Hdf5RldsDataConfig(
            repo_id = "sim_bimanual_assembly",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "sim_bimanual_assembly", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/sim_bimanual_assembly",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/sim_bimanual_assembly_pi05/",
            max_token_len = 96,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # q_network_config is byte-identical to the subtask_ar base's
        # q_network_config (224x224 images, no_state=True, predict_subtask_ar=True,
        # default paligemma backbone, no layernorm) — restated so the pretrained
        # checkpoint loads without any shape mismatch and the AR behavior is preserved.
        model_overrides = {
            "action_horizon": 60,
            "q_network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
        },
        # Bump the Best-of-N policy's action_horizon to match the 60-frame
        # data chunks. Without this the policy stays at the pretrain 50 and
        # value_function_objectives.py reshape (~lines 339, 507, 521) blows up
        # on the 60-frame CF candidates.
        policy_overrides = {
            "action_horizon": 60,
        },
        action_horizon = 60,
        num_train_steps = 20_000,
        save_interval = 10_000,
        plot_interval = 10_000,
        keep_period = 10_000,
        # Cosine decay 5e-6 -> 5e-7 over the 20k FT steps. FineTuneConfig wraps
        # this in an OffsetSchedule with offset=pretrained_step, so step 0 of the
        # cosine corresponds to the FT-start absolute step.
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 0, peak_lr = 5e-6, decay_steps = 20_000, decay_lr = 5e-7,
        ),
        num_val_trajectories = 3,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/sim_bimanual_assembly/validation_cache_dir_sim_bimanual_assembly_paligemma_cql_rlds_finetune_subtask_ar_final/",
        include_repos = (),
    ),
    FineTuneConfig(
        name = "real_shirt_hang_q_sarsa_finetune_task_description",
        data_factory = Hdf5RldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        model_overrides = {
            "action_horizon": 60,
            "network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 48,
                action_dim = 14,
                dtype = "float32",
                use_layernorm = True,
                no_state = True,
            ),
        },
        action_horizon = 60,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocoin/validation_cache_dir_real_shirt_hang_task_description/",
        num_val_trajectories = 2,
        include_repos = (),
        num_train_steps = 20_000,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        save_interval = 5000,
        plot_interval = 5000,
        keep_period = 5000,
    ),
    FineTuneConfig(
        name = "real_shirt_hang_q_sarsa_finetune_task_description_state",
        data_factory = Hdf5RldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        model_overrides = {
            "action_horizon": 60,
            "network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 48,
                action_dim = 14,
                dtype = "float32",
                use_layernorm = True,
                no_state = False,
            ),
        },
        action_horizon = 60,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocoin/validation_cache_dir_real_shirt_hang_task_description_state/",
        num_val_trajectories = 3,
        include_repos = (),
        num_train_steps = 20_000,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        save_interval = 5000,
        plot_interval = 5000,
        keep_period = 5000,
    ),
    # Same as real_shirt_hang_q_sarsa_finetune_task_description_state but configured
    # to fine-tune the gemma4 base (robocoin_bimanual_gemma4_q_sarsa_task_description):
    # 480x480 images via the gemma4_e2b backbone and the lower-shuffle / lower-parallelism
    # data pipeline gemma4 needs for its memory budget.
    FineTuneConfig(
        name = "real_shirt_hang_gemma4_q_sarsa_finetune_task_description_state_last_subtask_fix",
        data_factory = Hdf5RldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            image_size = (480, 480),
            shuffle_buffer_size = 7_500,
            num_parallel_reads = 4,
            num_parallel_calls = 4,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        # network_config mirrors the gemma4 base's network_config (paligemma_variant
        # "gemma4_e2b", 480x480 images) with no_state=False to keep the proprioceptive
        # state token.
        model_overrides = {
            "action_horizon": 60,
            "network_config": _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (480, 480),
                max_token_len = 48,
                action_dim = 14,
                dtype = "float32",
                paligemma_variant = "gemma4_e2b",
                use_layernorm = True,
                no_state = False,
            ),
        },
        action_horizon = 60,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocoin/validation_cache_dir_real_shirt_hang_gemma4_task_description_state/",
        num_val_trajectories = 3,
        include_repos = (),
        num_train_steps = 10_000,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        save_interval = 2000,
        plot_interval = 4000,
        keep_period = 2000,
    ),
    # Data-only fine-tune of robocasa_paligemma_q_sarsa_chunk_wise_delta_pretrain_atomic:
    # swaps datasets (PRETRAIN_ZERO_BASE_MOTION_ATOMIC → TARGET_COMPOSITE_JOINT) and
    # the norm-stats asset_id, leaving the network, optimizer, and data recipe
    # (critic_mode, chunk_wise_delta, quantile_norm, td_n, discount, native_fps,
    # interpolation_config = None, etc.) inherited from the base TrainConfig. Built
    # so the resulting critic shares action norm stats with robocasa_pi05_target_composite
    # (target__composite__joint), satisfying BestOfNWrapper's identical-norm-stats check
    # at serve time. include_repos lists the 3 composite repos so cache_val_episodes
    # caches exactly one trajectory per repo (num_val_trajectories == len(include_repos),
    # allow_duplicate_repos=False).
    FineTuneConfig(
        name = "robocasa_paligemma_q_sarsa_pretrain_atomic_to_target_composite",
        data_overrides = {
            "datasets": robocasa_datasets.TARGET_COMPOSITE_JOINT,
            "assets": AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__composite__joint",
            ),
        },
        num_train_steps = 8_000,
        save_interval = 2_000,
        plot_interval = 2_000,
        keep_period = 4_000,
        lr_schedule = _optimizer.ConstantSchedule(lr = 1e-6),
        num_val_trajectories = 3,
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_chunk_wise_delta_pretrain_atomic_to_target_composite/",
        include_repos = (
            "target__composite__prepare_coffee",
            "target__composite__weigh_ingredients",
            "target__composite__arrange_tea",
        ),
    ),
]

if len({c.name for c in _FINE_TUNE_CONFIGS}) != len(_FINE_TUNE_CONFIGS):
    raise ValueError("FineTuneConfig names must be unique.")
_FINE_TUNE_CONFIGS_DICT: dict[str, FineTuneConfig] = {c.name: c for c in _FINE_TUNE_CONFIGS}


def get_fine_tune_config(name: str) -> FineTuneConfig:
    """Get a FineTuneConfig by name."""
    if name not in _FINE_TUNE_CONFIGS_DICT:
        closest = difflib.get_close_matches(name, _FINE_TUNE_CONFIGS_DICT.keys(), n = 1, cutoff = 0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"FineTuneConfig '{name}' not found.{closest_str}")
    return _FINE_TUNE_CONFIGS_DICT[name]


# =============================================================================
# Train configs
# =============================================================================

# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora",
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    #
    # MLP D4RL configs.
    # Auto-detect state_dim and action_dim from Minari dataset.
    #
    *_make_antmaze_large_diverse_configs(),
    *_make_antmaze_large_diverse_v2_legacy_configs_safe(),
    *_make_pointmaze_large_configs(),
    TrainConfig(
        name="debug_mlp",
        model=mlp_config.MLPConfig(
            state_dim=29,
            action_dim=8,
            action_horizon=1,
            hidden_dims=(64, 64),
        ),
        data=FakeDataConfig(),
        batch_size=4,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_mlp",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
    # RoboCOIN Q(s,a) SARSA with bimanual dataset using the RLDS pipeline.
    TrainConfig(
        name="robocoin_bimanual_paligemma_q_sarsa",
        model=_value_function.SARSAValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                max_token_len=48,
                action_dim=14,
                dtype="float32",
                no_state=True,
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=RoboCoinRldsDataConfig(
            rlds_data_dir="gs://saksham-euw4/robocoin_bimanual/",
            assets=AssetsConfig(
                assets_dir="gs://saksham-euw4/robocoin_bimanual/",
                asset_id="norm_stats",
            ),
            datasets=(rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount=0.999,
            td_n=50,
            use_eef=True,
            use_quantile_norm=False,
            shuffle_buffer_size=50_000,
            mask_boundary_actions=False,
            replace_boundary_actions=True,
            use_chunk_wise_delta=False,
            state_dim=14,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=230_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=230_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=50_000,
        save_interval=50_000,
        fsdp_devices=16,
        action_horizon=50,
        num_val_trajectories=10,
        include_repos=("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
        validation_cache_dir="/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_50/",
    ),
    # Identical to robocoin_bimanual_paligemma_q_sarsa, but with quantile-norm + chunk-wise delta
    # actions and embodiment-wise norm stats.
    TrainConfig(
        name="robocoin_bimanual_paligemma_q_sarsa_chunk_wise_delta",
        model=_value_function.SARSAValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                max_token_len=48,
                action_dim=14,
                dtype="float32",
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=RoboCoinRldsDataConfig(
            rlds_data_dir="gs://saksham-euw4/robocoin_bimanual/",
            assets=AssetsConfig(
                assets_dir="gs://saksham-euw4/robocoin_bimanual/norm_stats",
                asset_id="embodiment_wise",
            ),
            datasets=(rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount=0.999,
            td_n=50,
            use_eef=True,
            use_quantile_norm=True,
            shuffle_buffer_size=50_000,
            mask_boundary_actions=False,
            replace_boundary_actions=False,
            use_chunk_wise_delta=True,
            state_dim=14,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=230_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=230_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=50_000,
        save_interval=50_000,
        fsdp_devices=16,
        action_horizon=50,
        num_val_trajectories=10,
        include_repos=("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
        validation_cache_dir="/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_50/",
    ),
    # From-scratch joint Q pretraining over the 49 RoboCasa zero-base-motion atomic
    # datasets (see robocasa_datasets.PRETRAIN_ZERO_BASE_MOTION_ATOMIC; weights ∝ step
    # counts so per-step draws are uniform across the union). Mirrors the network +
    # training recipe of robocoin_bimanual_paligemma_q_sarsa_chunk_wise_delta (SARSA +
    # PaliGemma Q net, chunk-wise-delta + quantile norm, PaliGemma weights). Native
    # 20 Hz (no FPS interpolation); action_horizon = td_n = 20 (= 1 sec at 20 Hz);
    # discount = 0.99 interpreted per native step. The joint asset_id below must be
    # produced by running compute_norm_stats.py over these 49 datasets first.
    TrainConfig(
        name = "robocasa_paligemma_q_sarsa_chunk_wise_delta_pretrain_atomic",
        model = _value_function.SARSAValueFunctionConfig(
            network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 48,
                action_dim = 14,
                dtype = "float32",
            ),
            head_config = _heads.RegressionHeadConfig(),
            action_horizon = 20,
        ),
        data = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = robocasa_datasets.PRETRAIN_ZERO_BASE_MOTION_ATOMIC,
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "pretrain__atomic__zero_base_motion_joint",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            use_all_cameras = True,
            image_size = 224,
            discount = 0.99,
            td_n = 20,
            native_fps = 20.0,
            interpolation_config = None,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
        ),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps = 10_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 10_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 4_000,
        save_interval = 2_000,
        keep_period = 4_000,
        fsdp_devices = 16,
        action_horizon = 20,
        num_val_trajectories = 5,
        include_repos = (
            "pretrain__atomic__turn_on_sink_faucet",
            "pretrain__atomic__turn_sink_spout",
            "pretrain__atomic__pick_place_counter_to_sink",
            "pretrain__atomic__pick_place_cabinet_to_counter",
            "pretrain__atomic__close_cabinet",
        ),
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_chunk_wise_delta_pretrain_atomic/",
    ),
    # Mirror of robocasa atomic config but pointed
    # at the 3 RoboCasa target composite datasets (see robocasa_datasets.TARGET_COMPOSITE_JOINT
    # for the step-weighted dataset list: prepare_coffee, weigh_ingredients, arrange_tea).
    # Same network / training recipe; differences are only on the data side.
    # prompt_mode defaults to "subtask": the composite subtask tracker rewrites the
    # per-step prompt to the active sub-instruction and emits steps_to_subtask_end —
    # all 3 datasets have registered CompositeSubtaskSpecs in robocasa_subtask_tracker.py.
    TrainConfig(
        name = "robocasa_paligemma_q_sarsa_chunk_wise_delta_target_composite",
        model = _value_function.SARSAValueFunctionConfig(
            network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 48,
                action_dim = 14,
                dtype = "float32",
            ),
            head_config = _heads.RegressionHeadConfig(),
            action_horizon = 20,
        ),
        data = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = robocasa_datasets.TARGET_COMPOSITE_JOINT,
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "pretrain__atomic__zero_base_motion_joint",
            ),
            critic_mode = True,
            bimanual_eef_layout = True,
            use_all_cameras = True,
            image_size = 224,
            discount = 0.99,
            td_n = 20,
            native_fps = 20.0,
            interpolation_config = None,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
        ),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps = 10_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 10_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 7_500,
        save_interval = 7_500,
        fsdp_devices = 16,
        action_horizon = 20,
        num_val_trajectories = 3,
        include_repos = (
            "target__composite__prepare_coffee",
            "target__composite__weigh_ingredients",
            "target__composite__arrange_tea",
        ),
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_chunk_wise_delta_target_composite/",
    ),
    # Pi-0.5 fine-tune from pi05_base on 3 RoboCasa composite tasks (target__composite__
    # prepare_coffee, weigh_ingredients, arrange_tea), step-weighted. Uses
    # prompt_mode="task_description" so the composite subtask tracker is bypassed and
    # the raw per-step language_instruction drives the prompt. Native 20 Hz, no FPS
    # interpolation; action_horizon = 20 (= 1 sec at 20 Hz). max_token_len = 48 chosen
    # to comfortably cover the composite language_instructions via PaligemmaTokenizer.
    # The joint asset_id below must be produced by running compute_norm_stats.py first.
    TrainConfig(
        name = "robocasa_pi05_target_composite",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 20,
            max_token_len = 48,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 0,
            action_dim_mask = (True,) * 7 + (False,) * 25,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = robocasa_datasets.TARGET_COMPOSITE_JOINT,
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__composite__joint",
            ),
            critic_mode = False,
            bimanual_eef_layout = True,
            use_all_cameras = True,
            image_size = 224,
            discount = 0.99,
            td_n = 20,
            native_fps = 20.0,
            interpolation_config = None,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            prompt_mode = "task_description",
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 70_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-5,
            decay_steps = 70_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        fsdp_devices = 16,
        action_horizon = 20,
    ),
    # Identical to robocasa_pi05_target_composite above, except prompt_mode is set
    # explicitly to "subtask": the composite subtask tracker rewrites the per-step
    # prompt to the active sub-instruction (and emits steps_to_subtask_end) instead
    # of using the raw task_description language_instruction.
    TrainConfig(
        name = "robocasa_pi05_target_composite_subtask",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 20,
            max_token_len = 48,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 0,
            action_dim_mask = (True,) * 7 + (False,) * 25,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = robocasa_datasets.TARGET_COMPOSITE_JOINT,
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__composite__joint",
            ),
            critic_mode = False,
            bimanual_eef_layout = True,
            use_all_cameras = True,
            image_size = 224,
            discount = 0.99,
            td_n = 20,
            native_fps = 20.0,
            interpolation_config = None,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            prompt_mode = "subtask",
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 70_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-5,
            decay_steps = 70_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        fsdp_devices = 16,
        action_horizon = 20,
    ),
    # Per-task from-scratch TrainConfigs. Same effective settings as applying the
    # robocasa FineTuneConfig on top of robocoin, except the training
    # is from scratch from PaliGemma weights (no RoboCOIN pretraining phase).
    TrainConfig(
        name="robocasa_paligemma_q_sarsa_chunk_wise_delta_coffee_setup_mug",
        model=_value_function.SARSAValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                max_token_len=48,
                action_dim=14,
                dtype="float32",
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=RLDSRoboCasaDataConfig(
            rlds_data_dir="gs://saksham-euw4/robocasa",
            datasets=(
                rlds_dataset.RLDSDataset(name = "target__atomic__coffee_setup_mug", version = "1.0.0", weight = 1.0),
            ),
            assets=AssetsConfig(
                assets_dir="gs://saksham-euw4/robocasa/norm_stats",
                asset_id="target__atomic__coffee_setup_mug",
            ),
            critic_mode=True,
            bimanual_eef_layout=True,
            image_size=224,
            discount=0.999,
            native_fps=20.0,
            interpolation_config=state_action_spaces.InterpolationConfig(
                target_fps=30.0, action_horizon_seconds=1.0,
            ),
            mask_boundary_actions=False,
            replace_boundary_actions=False,
            use_chunk_wise_delta=True,
            use_quantile_norm=True,
            shuffle_buffer_size=50_000,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=5_000,
        batch_size=256,
        lr_schedule=_optimizer.ConstantSchedule(lr=1e-6),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=2_500,
        save_interval=2_500,
        keep_period=2_500,
        fsdp_devices=16,
        action_horizon=50,
        num_val_trajectories=2,
        include_repos=(),
        validation_cache_dir="/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_chunk_wise_delta_coffee_setup_mug/",
    ),
    TrainConfig(
        name="robocasa_paligemma_q_sarsa_chunk_wise_delta_turn_on_sink_faucet",
        model=_value_function.SARSAValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                max_token_len=48,
                action_dim=14,
                dtype="float32",
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=RLDSRoboCasaDataConfig(
            rlds_data_dir="gs://saksham-euw4/robocasa",
            datasets=(
                rlds_dataset.RLDSDataset(name = "target__atomic__turn_on_sink_faucet", version = "1.0.0", weight = 1.0),
            ),
            assets=AssetsConfig(
                assets_dir="gs://saksham-euw4/robocasa/norm_stats",
                asset_id="target__atomic__turn_on_sink_faucet",
            ),
            critic_mode=True,
            bimanual_eef_layout=True,
            image_size=224,
            discount=0.999,
            native_fps=20.0,
            interpolation_config=state_action_spaces.InterpolationConfig(
                target_fps=30.0, action_horizon_seconds=1.0,
            ),
            mask_boundary_actions=False,
            replace_boundary_actions=False,
            use_chunk_wise_delta=True,
            use_quantile_norm=True,
            shuffle_buffer_size=50_000,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=5_000,
        batch_size=256,
        lr_schedule=_optimizer.ConstantSchedule(lr=1e-6),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=2_500,
        save_interval=2_500,
        keep_period=2_500,
        fsdp_devices=16,
        action_horizon=50,
        num_val_trajectories=2,
        include_repos=(),
        validation_cache_dir="/nfs/aidm_nfs/saksham3/robocasa/validation_cache_dir_chunk_wise_delta_turn_on_sink_faucet/",
    ),
    TrainConfig(
        name="robocoin_bimanual_paligemma_q_sarsa_task_description",
        model=_value_function.SARSAValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                max_token_len=96,
                action_dim=14,
                dtype="float32",
                use_layernorm=True,
            ),
            head_config=_heads.RegressionHeadConfig(),
            next_token_loss_weight=0.1,
        ),
        data=RoboCoinRldsDataConfig(
            rlds_data_dir="gs://saksham-euw4/robocoin_bimanual/",
            assets=AssetsConfig(
                assets_dir="gs://saksham-euw4/robocoin_bimanual/norm_stats",
                asset_id="embodiment_wise",
            ),
            datasets=(rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount=0.999,
            td_n=50,
            use_eef=True,
            use_quantile_norm=True,
            shuffle_buffer_size=50_000,
            mask_boundary_actions=False,
            replace_boundary_actions=False,
            use_chunk_wise_delta=True,
            state_dim=14,
            subtask_prompt_mode="task_description_predict_current_subtask",
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=230_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=230_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=50_000,
        save_interval=25_000,
        fsdp_devices=16,
        action_horizon=50,
        num_val_trajectories=10,
        include_repos=("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
        validation_cache_dir="/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_task_description/",
    ),
    TrainConfig(
        name="robocoin_bimanual_paligemma_q_sarsa_variable_horizon",
        model=_value_function.SARSAValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                max_token_len=48,
                action_dim=14,
                dtype="float32",
                no_state=True,
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=RoboCoinRldsDataConfig(
            rlds_data_dir="/data/group_data/rl/datasets/",
            assets=AssetsConfig(
                assets_dir="gs://saksham-euw4/robocoin_bimanual/norm_stats",
                asset_id=".",
            ),
            datasets=(rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount=0.999,
            td_n=50,
            use_eef=True,
            use_quantile_norm=False,
            shuffle_buffer_size=50_000,
            mask_boundary_actions=False,
            replace_boundary_actions=False,
            use_chunk_wise_delta=False,
            variable_horizon=True,
            state_dim=14,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=230_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=230_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=50_000,
        save_interval=50_000,
        fsdp_devices=16,
        action_horizon=50,
        num_val_trajectories=10,
        include_repos=("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
        validation_cache_dir="/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_variable_horizon/",
    ),
    # Direct training of paligemma Q(s,a) SARSA on real_hang (skips robocoin pretraining).
    # Mirrors the model + data settings of robocoin_bimanual_paligemma_q_sarsa
    # combined with the real_hang_finetune_q_sarsa overrides (subsample=True, action_horizon=30).
    TrainConfig(
        name="real_hang_paligemma_q_sarsa",
        model=_value_function.SARSAValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                max_token_len=48,
                action_dim=14,
                dtype="float32",
                no_state=True,
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=Hdf5RldsDataConfig(
            rlds_data_dir="gs://saksham-euw4/hdf5/real_hang_60_Hz",
            assets=AssetsConfig(
                assets_dir="gs://saksham-euw4/hdf5/real_hang_60_Hz",
                asset_id="norm_stats",
            ),
            datasets=(rlds_dataset.RLDSDataset(name = "real_hang", version = "1.0.0", weight = 1.0),),
            discount=0.999,
            td_n=60,
            use_eef=True,
            use_quantile_norm=False,
            shuffle_buffer_size=50_000,
            mask_boundary_actions=False,
            replace_boundary_actions=False,
            use_chunk_wise_delta=False,
            state_dim=14,
            subsample=True,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=25_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=25_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=5_000,
        save_interval=12_500,
        fsdp_devices=16,
        action_horizon=60,
        num_val_trajectories=2,
        include_repos=(),
        validation_cache_dir="/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_real_hang_q_sarsa/",
    ),
    # RoboCOIN Q(s,a) SARSA with Gemma 4 (E4B) backbone. Uses the Gemma-4 vision
    # encoder at 480x720 (30x45 patches → 10x15 pooled → 150 soft tokens/image)
    # instead of SigLIP.
    TrainConfig(
        name="robocoin_bimanual_gemma4_q_sarsa_task_description",
        model=_value_function.SARSAValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(480, 480),
                max_token_len=96,
                action_dim=14,
                dtype="float32",
                paligemma_variant="gemma4_e2b",
                use_layernorm=True,
            ),
            head_config=_heads.RegressionHeadConfig(),
            next_token_loss_weight=0.1,
        ),
        data=RoboCoinRldsDataConfig(
            rlds_data_dir="gs://saksham-euw4/robocoin_bimanual_unresized",
            assets=AssetsConfig(
                assets_dir="gs://saksham-euw4/robocoin_bimanual/norm_stats",
                asset_id="embodiment_wise",
            ),
            datasets=(rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount=0.999,
            td_n=50,
            use_eef=True,
            use_quantile_norm=True,
            image_size=(480, 480),
            shuffle_buffer_size=7_500,
            num_parallel_reads=4,
            num_parallel_calls=4,
            mask_boundary_actions=False,
            replace_boundary_actions=False,
            use_chunk_wise_delta=True,
            state_dim=14,
            subtask_prompt_mode="task_description_predict_current_subtask",
        ),
        weight_loader=weight_loaders.Gemma4WeightLoader(
            checkpoint_path="gs://gemma-data/checkpoints/gemma4-e2b-pt",
        ),
        num_train_steps=230_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5000,
            peak_lr=1e-5,
            decay_steps=230_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=50_000,
        save_interval=25_000,
        fsdp_devices=16,
        action_horizon=50,
        num_val_trajectories=10,
        include_repos=("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
        validation_cache_dir="/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_gemma4_task_description/",
        backbone_variant="gemma4",
    ),
    # RoboCOIN CQL Q(s,a) with pi-0.5 (PaliGemma) backbone, Best-of-N policy wrapper
    TrainConfig(
        name = "robocoin_bimanual_paligemma_cql_rlds",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.1,
            action_horizon = 50,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 50,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        policy_extraction = _policy_extraction.NoopPolicyConfig(),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocoin_bimanual",
            # rlds_data_dir="/data/group_data/rl/datasets/",
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocoin_bimanual/norm_stats",
                # assets_dir="/data/group_data/rl/saksham3/robocoin/norm_stats",
                asset_id = "embodiment_wise",
            ),
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount = 0.999,
            td_n = 50,
            use_eef = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/robocoin_bimanual_pi05_rlds",
            # counterfactual_action_store_dir="/data/group_data/rl/saksham3/robocoin/cached_actions/pi05_finetune_8",
            state_dim = 14,
            max_token_len = 96,
            subtask_prompt_mode = "task_description_predict_current_subtask",
        ),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 50,
        num_val_trajectories = 10,
        include_repos = ("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_cql_rlds/",
    ),
    # Copy of robocoin_bimanual_paligemma_cql_rlds with predict_subtask_ar=True:
    # the subtask suffix stays visible to state/action/CLS queries at its natural
    # RoPE positions (no suffix blocking, no position shift) while remaining
    # causal for the next-token objective. Only the name and the flag differ.
    TrainConfig(
        name = "robocoin_bimanual_paligemma_cql_rlds_subtask_ar",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.1,
            action_horizon = 50,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 50,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        policy_extraction = _policy_extraction.NoopPolicyConfig(),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocoin_bimanual",
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocoin_bimanual/norm_stats",
                asset_id = "embodiment_wise",
            ),
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount = 0.999,
            td_n = 50,
            use_eef = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/robocoin_bimanual_pi05_rlds",
            state_dim = 14,
            max_token_len = 96,
            subtask_prompt_mode = "task_description_predict_current_subtask",
        ),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 50,
        num_val_trajectories = 10,
        include_repos = ("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_cql_rlds/",
    ),
    # Copy of robocoin_bimanual_paligemma_cql_rlds_subtask_ar but with a from-scratch
    # gemma_300m LLM backbone and the SigLIP vision tower loaded from pretrained PaliGemma
    # (PaliGemmaSiglipOnlyWeightLoader loads only the img subtree; the LLM stays random).
    # Only the name, paligemma_variant, and weight_loader differ.
    TrainConfig(
        name = "robocoin_bimanual_paligemma_cql_rlds_subtask_ar_gemma300m_scratch",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
                paligemma_variant = "gemma_300m",
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.1,
            action_horizon = 50,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 50,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        policy_extraction = _policy_extraction.NoopPolicyConfig(),
        weight_loader = weight_loaders.PaliGemmaSiglipOnlyWeightLoader(),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocoin_bimanual",
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocoin_bimanual/norm_stats",
                asset_id = "embodiment_wise",
            ),
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount = 0.999,
            td_n = 50,
            use_eef = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/robocoin_bimanual_pi05_rlds",
            state_dim = 14,
            max_token_len = 96,
            subtask_prompt_mode = "task_description_predict_current_subtask",
        ),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 50,
        num_val_trajectories = 10,
        include_repos = ("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_cql_rlds_subtask_ar/",
    ),
    # Copy of robocoin_bimanual_paligemma_cql_rlds with
    # predict_subtask_ar=True and variable_horizon=True.
    # save_interval is 25k; the validation cache is shared with
    # robocoin_bimanual_paligemma_cql_rlds_subtask_ar (cached val episodes hold
    # obs/action/mc_return, which neither flag changes).
    TrainConfig(
        name = "robocoin_bimanual_paligemma_cql_rlds_variable_horizon_subtask_ar",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.1,
            action_horizon = 50,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 50,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        policy_extraction = _policy_extraction.NoopPolicyConfig(),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocoin_bimanual",
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocoin_bimanual/norm_stats",
                asset_id = "embodiment_wise",
            ),
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount = 0.999,
            td_n = 50,
            use_eef = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            variable_horizon = True,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/robocoin_bimanual_pi05_rlds",
            state_dim = 14,
            max_token_len = 96,
            subtask_prompt_mode = "task_description_predict_current_subtask",
        ),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 25_000,
        fsdp_devices = 16,
        action_horizon = 50,
        num_val_trajectories = 10,
        include_repos = ("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_cql_rlds_subtask_ar/",
    ),
    # Copy of robocoin_bimanual_paligemma_cql_rlds_variable_horizon_subtask_ar with lower_action_horizon=25.
    TrainConfig(
        name = "robocoin_bimanual_paligemma_cql_rlds_variable_horizon_subtask_ar_lb25",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                predict_subtask_ar = True,
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.1,
            action_horizon = 50,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 50,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        policy_extraction = _policy_extraction.NoopPolicyConfig(),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocoin_bimanual",
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocoin_bimanual/norm_stats",
                asset_id = "embodiment_wise",
            ),
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount = 0.999,
            td_n = 50,
            use_eef = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            variable_horizon = True,
            lower_action_horizon = 25,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/robocoin_bimanual_pi05_rlds",
            state_dim = 14,
            max_token_len = 96,
            subtask_prompt_mode = "task_description_predict_current_subtask",
        ),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 25_000,
        fsdp_devices = 16,
        action_horizon = 50,
        num_val_trajectories = 10,
        include_repos = ("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_cql_rlds_subtask_ar/",
    ),
    # Copy of robocoin_bimanual_paligemma_cql_rlds with the next-token-prediction
    # auxiliary loss disabled (next_token_loss_weight=0.0). Only the name,
    # validation_cache_dir, and the ntp weight differ from the base config.
    TrainConfig(
        name = "robocoin_bimanual_paligemma_cql_rlds_no_ntp",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.0,
            action_horizon = 50,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 50,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        policy_extraction = _policy_extraction.NoopPolicyConfig(),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocoin_bimanual",
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocoin_bimanual/norm_stats",
                asset_id = "embodiment_wise",
            ),
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount = 0.999,
            td_n = 50,
            use_eef = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/robocoin_bimanual_pi05_rlds",
            state_dim = 14,
            max_token_len = 96,
            subtask_prompt_mode = "task_description_predict_current_subtask",
        ),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 50,
        num_val_trajectories = 10,
        include_repos = ("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_cql_rlds/",
    ),
    # From-scratch baseline on the real_shirt_hang HDF5 dataset: gemma_300m
    # backbone, random weights (NoOpWeightLoader, both Gemma LLM and SigLIP tower
    # randomly initialized). Data + counterfactual-action store taken from
    # real_shirt_hang_paligemma_cql_rlds_finetune_task_description_final
    # (Hdf5RldsDataConfig, real_shirt_hang, td_n=60, subsample, 60-frame chunks).
    TrainConfig(
        name = "real_shirt_hang_paligemma_cql_rlds_gemma300m_scratch",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                paligemma_variant = "gemma_300m",
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.0,
            action_horizon = 60,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 60,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        policy_extraction = _policy_extraction.NoopPolicyConfig(),
        weight_loader = weight_loaders.PaliGemmaSiglipOnlyWeightLoader(),
        data = Hdf5RldsDataConfig(
            repo_id = "real_shirt_hang",
            rlds_data_dir = "/data/group_data/rl/saksham3/hdf5",
            datasets = (rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 100_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = "/data/group_data/rl/saksham3/robocoin/cached_actions/real_shirt_hang_pi05",
            max_token_len = 96,
            prompt_mode = "task_description_predict_current_subtask",
        ),
        num_train_steps = 20_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 20_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 60,
        num_val_trajectories = 3,
        include_repos = (),
        validation_cache_dir = "/data/user_data/saksham3/real_shirt_hang/validation_cache/gemma300m_scratch/",
    ),
    # Same as real_shirt_hang_paligemma_cql_rlds_gemma300m_scratch but with the
    # default prompt_mode="subtask": the per-frame current subtask is fed directly
    # as the text prompt (no task-description prefix, no next-token prediction —
    # plain regression head conditioned on the subtask).
    TrainConfig(
        name = "real_shirt_hang_paligemma_cql_rlds_gemma300m_scratch_subtask",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
                paligemma_variant = "gemma_300m",
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            action_horizon = 60,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 60,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        policy_extraction = _policy_extraction.NoopPolicyConfig(),
        weight_loader = weight_loaders.PaliGemmaSiglipOnlyWeightLoader(),
        data = Hdf5RldsDataConfig(
            repo_id = "real_shirt_hang",
            rlds_data_dir = "/data/group_data/rl/saksham3/hdf5",
            datasets = (rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            state_dim = 14,
            critic_mode = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 100_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            subsample = True,
            counterfactual_action_store_dir = "/data/group_data/rl/saksham3/robocoin/cached_actions/real_shirt_hang_pi05",
            max_token_len = 96,
            prompt_mode = "subtask",
        ),
        num_train_steps = 20_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 20_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 60,
        num_val_trajectories = 3,
        include_repos = (),
        validation_cache_dir = "/data/user_data/saksham3/real_shirt_hang/validation_cache/gemma300m_scratch_subtask/",
    ),
    # Same as robocoin_bimanual_paligemma_cql_rlds but feeds the sampled subtask
    # directly as the text prompt (subtask_prompt_mode="subtask_only", the field
    # default, set explicitly) and disables the next-token (current-subtask)
    # prediction loss (next_token_loss_weight=0.0) — pure CQL + Best-of-N TD value
    # learning over cached counterfactual actions.
    TrainConfig(
        name = "robocoin_bimanual_paligemma_cql_rlds_subtask_no_ntp",
        model = _value_function.CQLValueFunctionConfig(
            q_network_config = _paligemma_network.PaliGemmaNetworkConfig(
                state_dim = 14,
                num_cameras = 3,
                image_size = (224, 224),
                max_token_len = 96,
                action_dim = 14,
                dtype = "float32",
                no_state = True,
            ),
            q_head_config = _heads.RegressionHeadConfig(),
            next_token_loss_weight = 0.0,
            action_horizon = 50,
            discount = 0.999,
            tau = 0.005,
            action_bounds = ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha = 0.0,
        ),
        policy = _best_of_n.BestOfNWrapperConfig(
            action_dim = 14,
            action_horizon = 50,
            base_model_config = None,
            num_samples = 8,
            use_target_value = True,
        ),
        policy_extraction = _policy_extraction.NoopPolicyConfig(),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocoin_bimanual",
            # rlds_data_dir="/data/group_data/rl/datasets/",
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocoin_bimanual/norm_stats",
                # assets_dir="/data/group_data/rl/saksham3/robocoin/norm_stats",
                asset_id = "embodiment_wise",
            ),
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount = 0.999,
            td_n = 50,
            use_eef = True,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            counterfactual_action_store_dir = "gs://saksham-euw4/robocoin/cached_actions/robocoin_bimanual_pi05_rlds",
            # counterfactual_action_store_dir="/data/group_data/rl/saksham3/robocoin/cached_actions/pi05_finetune_8",
            state_dim = 14,
            max_token_len = 96,
            subtask_prompt_mode = "subtask_only",
        ),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-6,
        ),
        optimizer = _optimizer.AdamW(weight_decay = 1e-6),
        num_workers = 0,
        log_interval = 100,
        plot_interval = 50_000,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 50,
        num_val_trajectories = 10,
        include_repos = ("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
        validation_cache_dir = "/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_cql_rlds_subtask/",
    ),
    # Gemma4 mirror of robocoin_bimanual_paligemma_cql_rlds: same CQL + Best-of-N
    # setup, swapped to the gemma4_e2b backbone (480x480 unresized images, Gemma4
    # weight loader, smaller shuffle buffer for memory).
    TrainConfig(
        name="robocoin_bimanual_gemma4_cql_rlds",
        model=_value_function.CQLValueFunctionConfig(
            q_network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(480, 480),
                max_token_len=96,
                action_dim=14,
                dtype="float32",
                paligemma_variant="gemma4_e2b",
                use_layernorm=True,
                predict_subtask_ar=True,
                no_state=True,
            ),
            q_head_config=_heads.RegressionHeadConfig(),
            next_token_loss_weight=0.1,
            action_horizon=50,
            discount=0.999,
            tau=0.005,
            action_bounds=ActionBounds.from_uniform(-1.25, 1.25, action_dim = 14, is_normalized = True),
            cql_alpha=0.0,
        ),
        policy=_best_of_n.BestOfNWrapperConfig(
            action_dim=14,
            action_horizon=50,
            base_model_config=None,
            num_samples=8,
            use_target_value=True,
        ),
        policy_extraction=_policy_extraction.NoopPolicyConfig(),
        weight_loader=weight_loaders.Gemma4WeightLoader(
            checkpoint_path="gs://gemma-data/checkpoints/gemma4-e2b-pt",
        ),
        data=RoboCoinRldsDataConfig(
            rlds_data_dir="gs://saksham-euw4/robocoin_bimanual_unresized",
            assets=AssetsConfig(
                assets_dir="gs://saksham-euw4/robocoin_bimanual/norm_stats",
                asset_id="embodiment_wise",
            ),
            datasets=(rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount=0.999,
            td_n=50,
            use_eef=True,
            use_chunk_wise_delta=True,
            use_quantile_norm=True,
            image_size=(480, 480),
            shuffle_buffer_size=7_500,
            num_parallel_reads=4,
            num_parallel_calls=4,
            mask_boundary_actions=False,
            replace_boundary_actions=False,
            counterfactual_action_store_dir="gs://saksham-euw4/robocoin/cached_actions/robocoin_bimanual_pi05_rlds_gemma4",
            state_dim=14,
            max_token_len=96,
            subtask_prompt_mode="task_description_predict_current_subtask",
        ),
        num_train_steps=230_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=230_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=50_000,
        save_interval=50_000,
        fsdp_devices=16,
        action_horizon=50,
        num_val_trajectories=10,
        include_repos=("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning", "RoboCOIN/R1_Lite_place_the_dress_shirt_on_the_hanger", "RoboCOIN/Split_aloha_pour_tea"),
        validation_cache_dir="/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache_gemma4_cql_rlds/",
        backbone_variant="gemma4",
    ),
    # =========================================================================
    # Reference: CQL + Best-of-N config (from main branch, cosmos backbone).
    # Requires: best_of_n.py, action_bounds.py, rlds_dataset.py,
    #           robocoin_rlds_dataset.py, counterfactual_action_store.py,
    #           cosmos network, CosmosValueWeightLoader.
    # =========================================================================
    # TrainConfig(
    #     name="cosmos_robocoin_td_learning_hl_gauss",
    #     model=_value_function.CQLValueFunctionConfig(
    #         q_network_config=_cosmos_network.CosmosValueNetworkConfig(
    #             image_keys=("cam_0", "cam_1", "cam_2"),
    #             image_height=224,
    #             image_width=224,
    #             cosmos_kwargs_path="...",
    #             uncond_text_embeddings_path="...",
    #             action_conditioned=True,
    #             action_dim=14,
    #             action_horizon=30,
    #             fixed_timestep=0.0,
    #         ),
    #         q_head_config=_heads.CategoricalHeadConfig(
    #             v_min=-500.0,
    #             v_max=0.0,
    #             num_bins=128,
    #         ),
    #         discount=0.998,
    #         tau=0.005,
    #         action_bounds=ActionBounds.from_uniform(-1.0, 1.0, action_dim=14, is_normalized=True),
    #         cql_alpha=0.0,
    #     ),
    #     policy=_best_of_n.BestOfNWrapperConfig(
    #         action_dim=14,
    #         action_horizon=30,
    #         base_model_config=None,
    #         use_target_value=True,
    #     ),
    #     policy_extraction=_policy_extraction.NoopPolicyExtractionConfig(),
    #     weight_loader=weight_loaders.CosmosValueWeightLoader("..."),
    #     data=RoboCoinRldsDataConfig(
    #         rlds_data_dir="...",
    #         datasets=(rlds_dataset.RLDSDataset(name="robocoin", version="1.0.0", weight=1.0),),
    #         discount=0.998,
    #         reward_bias=-1.0,
    #         use_eef=True,
    #         counterfactual_action_store_dir="...",
    #         max_num_demos=1,
    #     ),
    #     num_train_steps=500_000,
    #     batch_size=1,
    #     shuffle_buffer_size=5_000,
    #     lr_schedule=_optimizer.CosineDecaySchedule(
    #         peak_lr=5e-5,
    #         decay_steps=500_000,
    #         decay_lr=1e-6,
    #         warmup_steps=1_000,
    #     ),
    #     optimizer=_optimizer.SGD(momentum=0.0),
    #     freeze_filter=_nnx_utils.PathRegex(".*vae.*"),
    #     plot_interval=10_000,
    #     num_workers=0,
    #     fsdp_devices=1,
    # ),
    TrainConfig(
        name = "robocoin_bimanual_pi05_rlds",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 50,
            max_token_len = 48,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 14,
            action_dim_mask = (False,) * 14 + (True,) * 14 + (False,) * 4,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = RoboCoinRldsDataConfig(
            rlds_data_dir = "/data/group_data/rl/datasets",
            datasets = (rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocoin_bimanual/norm_stats",
                # assets_dir = "/data/group_data/rl/saksham3/robocoin/norm_stats",
                asset_id = "embodiment_wise",
            ),
            discount = 0.999,
            td_n = 50,
            use_eef = True,
            critic_mode = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            filter_n = 5,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            replace_boundary_actions = False,
            state_dim = 14,
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 230_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 1e-5,
            decay_steps = 230_000,
            decay_lr = 1e-5,
        ),
        optimizer = _optimizer.AdamW(weight_decay=1e-6),
        num_workers = 0,
        log_interval = 100,
        save_interval = 50_000,
        fsdp_devices = 16,
        action_horizon = 50,
    ),
    # π-0.5 RoboCasa fine-tune: action loss is gated to the first 7 right-arm dims
    # (eef_pos:3, eef_rot:3, gripper:1); the remaining 25 dims are masked out.
    # `_gpu` variant uses cluster paths; the no-suffix variant below mirrors it with GCS
    # paths for TPU runs.
    TrainConfig(
        name = "robocasa_pi05_finetune_gpu",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 50,
            max_token_len = 48,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 0,
            # 32 = 7 (right arm, real) + 25 (masked).
            action_dim_mask = (True,) * 7 + (False,) * 25,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = RLDSRoboCasaDataConfig(
            rlds_data_dir = "/data/group_data/rl/datasets/robocasa_rlds",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__close_blender_lid", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "/data/group_data/rl/saksham3/datasets/robocasa/norm_stats",
                asset_id = "target__atomic__close_blender_lid",
            ),
            critic_mode = False,
            bimanual_eef_layout = True,
            image_size = 224,
            # FPS interpolation: 20 Hz native → 30 Hz target (matches the π-0.5 base
            # policy's training rate). action_horizon_seconds=1.0 → 30 valid frames,
            # which matches Pi0Config.action_horizon=30 above.
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            shuffle_buffer_size = 100_000,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            mask_boundary_actions = False,
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("/data/group_data/rl/saksham3/pi05_base_params/params/"),
        num_train_steps = 50_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 5e-5,
            decay_steps = 50_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 10_000,
        fsdp_devices = 4,
        action_horizon = 50,
    ),
    # GCS-paths mirror of robocasa_pi05_finetune_gpu for TPU runs.
    TrainConfig(
        name = "robocasa_pi05_finetune_close_blender_lid",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 50,
            max_token_len = 48,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 0,
            action_dim_mask = (True,) * 7 + (False,) * 25,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__close_blender_lid", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__close_blender_lid",
            ),
            critic_mode = False,
            bimanual_eef_layout = True,
            image_size = 224,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            shuffle_buffer_size = 50_000,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            mask_boundary_actions = False,
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 70_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 5e-5,
            decay_steps = 70_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 10_000,
        fsdp_devices = 16,
        action_horizon = 50,
    ),
    # `use_all_cameras=True` mirror of robocasa_pi05_finetune_close_blender_lid: routes all three
    # RoboCasa cameras into the bimanual image slots (left/top → base_0_rgb, right/top →
    # left_wrist_0_rgb, wrist_camera → right_wrist_0_rgb) instead of masking the third.
    TrainConfig(
        name = "robocasa_pi05_finetune_close_blender_lid_use_all_cameras",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 50,
            max_token_len = 48,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 0,
            action_dim_mask = (True,) * 7 + (False,) * 25,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__close_blender_lid", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__close_blender_lid",
            ),
            critic_mode = False,
            bimanual_eef_layout = True,
            use_all_cameras = True,
            image_size = 224,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            shuffle_buffer_size = 50_000,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            mask_boundary_actions = False,
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 70_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 5e-5,
            decay_steps = 70_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 10_000,
        fsdp_devices = 16,
        action_horizon = 50,
    ),
    # Per-task variants of robocasa_pi05_finetune. Only the dataset name and
    # asset_id differ from the parent.
    TrainConfig(
        name = "robocasa_pi05_finetune_coffee_setup_mug",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 50,
            max_token_len = 48,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 0,
            action_dim_mask = (True,) * 7 + (False,) * 25,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__coffee_setup_mug", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__coffee_setup_mug",
            ),
            critic_mode = False,
            bimanual_eef_layout = True,
            image_size = 224,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            shuffle_buffer_size = 50_000,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            mask_boundary_actions = False,
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 70_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 5e-5,
            decay_steps = 70_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 10_000,
        fsdp_devices = 16,
        action_horizon = 50,
    ),
    TrainConfig(
        name = "robocasa_pi05_finetune_open_cabinet",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 50,
            max_token_len = 48,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 0,
            action_dim_mask = (True,) * 7 + (False,) * 25,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__open_cabinet", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__open_cabinet",
            ),
            critic_mode = False,
            bimanual_eef_layout = True,
            image_size = 224,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            shuffle_buffer_size = 50_000,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            mask_boundary_actions = False,
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 70_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 5e-5,
            decay_steps = 70_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 10_000,
        fsdp_devices = 16,
        action_horizon = 50,
    ),
    TrainConfig(
        name = "robocasa_pi05_finetune_close_fridge",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 50,
            max_token_len = 48,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 0,
            action_dim_mask = (True,) * 7 + (False,) * 25,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__close_fridge", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__close_fridge",
            ),
            critic_mode = False,
            bimanual_eef_layout = True,
            image_size = 224,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            shuffle_buffer_size = 50_000,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            mask_boundary_actions = False,
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 70_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 5e-5,
            decay_steps = 70_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 10_000,
        fsdp_devices = 16,
        action_horizon = 50,
    ),
    TrainConfig(
        name = "robocasa_pi05_finetune_pick_place_sink_to_counter",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 50,
            max_token_len = 48,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 0,
            action_dim_mask = (True,) * 7 + (False,) * 25,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__pick_place_sink_to_counter", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__pick_place_sink_to_counter",
            ),
            critic_mode = False,
            bimanual_eef_layout = True,
            image_size = 224,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            shuffle_buffer_size = 50_000,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            mask_boundary_actions = False,
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 70_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 5e-5,
            decay_steps = 70_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 10_000,
        fsdp_devices = 16,
        action_horizon = 50,
    ),
    TrainConfig(
        name = "robocasa_pi05_finetune_pick_place_toaster_to_counter",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 50,
            max_token_len = 48,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 0,
            action_dim_mask = (True,) * 7 + (False,) * 25,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__pick_place_toaster_to_counter", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__pick_place_toaster_to_counter",
            ),
            critic_mode = False,
            bimanual_eef_layout = True,
            image_size = 224,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            shuffle_buffer_size = 50_000,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            mask_boundary_actions = False,
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 70_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 5e-5,
            decay_steps = 70_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 10_000,
        fsdp_devices = 16,
        action_horizon = 50,
    ),
    TrainConfig(
        name = "robocasa_pi05_finetune_turn_on_sink_faucet",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 50,
            max_token_len = 48,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 0,
            action_dim_mask = (True,) * 7 + (False,) * 25,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = RLDSRoboCasaDataConfig(
            rlds_data_dir = "gs://saksham-euw4/robocasa",
            datasets = (
                rlds_dataset.RLDSDataset(name = "target__atomic__turn_on_sink_faucet", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/robocasa/norm_stats",
                asset_id = "target__atomic__turn_on_sink_faucet",
            ),
            critic_mode = False,
            bimanual_eef_layout = True,
            image_size = 224,
            native_fps = 20.0,
            interpolation_config = state_action_spaces.InterpolationConfig(
                target_fps = 30.0, action_horizon_seconds = 1.0,
            ),
            shuffle_buffer_size = 50_000,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            mask_boundary_actions = False,
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 70_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1_000,
            peak_lr = 5e-5,
            decay_steps = 70_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 10_000,
        fsdp_devices = 16,
        action_horizon = 50,
    ),
    # Pi-0.5 fine-tune on the sim_bimanual_assembly HDF5 dataset.
    TrainConfig(
        name = "sim_bimanual_assembly_pi05",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 60,
            max_token_len = 96,
            pi05 = True,
            discrete_state_input = True,
            action_dim_offset = 14,
            action_dim_mask = (False,) * 14 + (True,) * 14 + (False,) * 4,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = Hdf5RldsDataConfig(
            repo_id = "sim_bimanual_assembly",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "sim_bimanual_assembly", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/sim_bimanual_assembly",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            critic_mode = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            filter_n = 8,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            state_dim = 14,
            subsample = False,
            prompt_mode = "task_description",
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 200_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-5,
            decay_steps = 200_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 20_000,
        fsdp_devices = 16,
        action_horizon = 60,
    ),
    # Identical to sim_bimanual_assembly_pi05 but with filter_intervention=True.
    TrainConfig(
        name = "sim_bimanual_assembly_filter_pi05",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 60,
            max_token_len = 96,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 14,
            action_dim_mask = (False,) * 14 + (True,) * 14 + (False,) * 4,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = Hdf5RldsDataConfig(
            repo_id = "sim_bimanual_assembly",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "sim_bimanual_assembly", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/sim_bimanual_assembly",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            critic_mode = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            filter_intervention = True,
            filter_repo_index = (0, 5, 7),
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            state_dim = 14,
            subsample = False,
            prompt_mode = "task_description",
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 100_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-5,
            decay_steps = 100_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 10_000,
        fsdp_devices = 16,
        action_horizon = 60,
    ),
    # Identical to sim_bimanual_assembly_pi05 but with subsample=True (and a
    # 100k-step training/LR schedule, keep_period=10k).
    TrainConfig(
        name = "sim_bimanual_assembly_pi05_subsample",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 60,
            max_token_len = 96,
            pi05 = True,
            discrete_state_input = True,
            action_dim_offset = 14,
            action_dim_mask = (False,) * 14 + (True,) * 14 + (False,) * 4,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = Hdf5RldsDataConfig(
            repo_id = "sim_bimanual_assembly",
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (
                rlds_dataset.RLDSDataset(name = "sim_bimanual_assembly", version = "1.0.0", weight = 1.0),
            ),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/sim_bimanual_assembly",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            critic_mode = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            filter_n = 4,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            state_dim = 14,
            subsample = True,
            prompt_mode = "task_description",
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 100_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-5,
            decay_steps = 100_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 10_000,
        fsdp_devices = 16,
        action_horizon = 60,
    ),
    # Pi-0.5 on the real_shirt_hang dataset.
    TrainConfig(
        name = "real_shirt_hang_pi05",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 60,
            max_token_len = 96,
            pi05 = True,
            discrete_state_input = True,
            action_dim_offset = 14,
            action_dim_mask = (False,) * 14 + (True,) * 14 + (False,) * 4,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = Hdf5RldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            critic_mode = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            filter_n = 8,
            # filter_intervention = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            state_dim = 14,
            subsample = False,
            prompt_mode = "task_description",
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 200_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-5,
            decay_steps = 200_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 20_000,
        fsdp_devices = 16,
        action_horizon = 60,
    ),
    # Pi-0.5 fine-tune on the realworld_xarm_packing (LeRobot-built) dataset.
    TrainConfig(
        name = "realworld_xarm_packing_pi05",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 60,
            max_token_len = 128,
            pi05 = True,
            discrete_state_input = True,
            action_dim_offset = 14,
            action_dim_mask = (False,) * 14 + (True,) * 14 + (False,) * 4,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = LeRobotRldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/datasets/realworld_xarm_packing",
            datasets = (rlds_dataset.RLDSDataset(name = "realworld_xarm_packing", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/datasets/realworld_xarm_packing",
                asset_id = "norm_stats",
            ),
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            filter_n = 8,
            shuffle_buffer_size = 50_000,
            prompt_mode = "task_description",
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 70_000,
        batch_size = 256,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-5,
            decay_steps = 70_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 5_000,
        keep_period = 25_000,
        fsdp_devices = 16,
        action_horizon = 60,
    ),
    # Identical to real_shirt_hang_pi05 but with filter_intervention=True.
    TrainConfig(
        name = "real_shirt_hang_filter_pi05",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 60,
            max_token_len = 96,
            pi05 = True,
            discrete_state_input = False,
            action_dim_offset = 14,
            action_dim_mask = (False,) * 14 + (True,) * 14 + (False,) * 4,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = Hdf5RldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            critic_mode = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            filter_intervention = True,
            filter_repo_index = (0, 1, 2, 3, 10, 11),
            shuffle_buffer_size = 50_000,
            num_parallel_reads = 4,
            num_parallel_calls = 4,
            mask_boundary_actions = False,
            state_dim = 14,
            subsample = False,
            prompt_mode = "task_description",
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 100_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-5,
            decay_steps = 100_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 20_000,
        fsdp_devices = 16,
        action_horizon = 60,
    ),
    # Identical to real_shirt_hang_pi05 but with subsample=True (and a
    # 100k-step training/LR schedule, keep_period=10k).
    TrainConfig(
        name = "real_shirt_hang_pi05_subsample",
        model = pi0_config.Pi0Config(
            paligemma_variant = "gemma_2b",
            action_expert_variant = "gemma_300m",
            action_dim = 32,
            action_horizon = 60,
            max_token_len = 96,
            pi05 = True,
            discrete_state_input = True,
            action_dim_offset = 14,
            action_dim_mask = (False,) * 14 + (True,) * 14 + (False,) * 4,
            pad_state_to_action_dim = False,
            dtype = "float32",
        ),
        data = Hdf5RldsDataConfig(
            rlds_data_dir = "gs://saksham-euw4/hdf5",
            datasets = (rlds_dataset.RLDSDataset(name = "real_shirt_hang", version = "1.0.0", weight = 1.0),),
            assets = AssetsConfig(
                assets_dir = "gs://saksham-euw4/hdf5/real_shirt_hang",
                asset_id = "norm_stats",
            ),
            discount = 0.999,
            td_n = 60,
            use_eef = True,
            critic_mode = False,
            use_chunk_wise_delta = True,
            use_quantile_norm = True,
            filter_n = 4,
            # filter_intervention = True,
            shuffle_buffer_size = 50_000,
            mask_boundary_actions = False,
            state_dim = 14,
            subsample = True,
            prompt_mode = "task_description",
        ),
        weight_loader = weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps = 100_000,
        batch_size = 128,
        lr_schedule = _optimizer.CosineDecaySchedule(
            warmup_steps = 1000,
            peak_lr = 5e-5,
            decay_steps = 100_000,
            decay_lr = 5e-6,
        ),
        optimizer = _optimizer.AdamW(),
        num_workers = 0,
        log_interval = 100,
        save_interval = 10_000,
        keep_period = 10_000,
        fsdp_devices = 16,
        action_horizon = 60,
    ),
    # Same as robocoin_bimanual_pi05_no_mask but with state_dim=16 and discrete_state_input=True
    TrainConfig(
        name="robocoin_bimanual_pi05_state",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            action_dim=32,
            action_horizon=50,
            max_token_len=96,
            pi05=True,
            discrete_state_input=True,
            action_dim_offset=14,
            action_dim_mask=(False,) * 14 + (True,) * 14 + (False,) * 4,
            dtype="float32",
        ),
        data=RoboCoinRldsDataConfig(
            rlds_data_dir="/data/group_data/rl/datasets/",
            assets=AssetsConfig(
                assets_dir="/data/group_data/rl/saksham3/robocoin/norm_stats",
                asset_id="embodiment_wise",
            ),
            datasets=(rlds_dataset.RLDSDataset(name = "robocoin", version = "1.0.0", weight = 1.0),),
            discount=0.999,
            td_n=50,
            use_eef=True,
            critic_mode=False,
            use_chunk_wise_delta=True,
            use_quantile_norm=True,
            filter_n=5,
            shuffle_buffer_size=200_000,
            mask_boundary_actions=False,
            state_dim=16,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=50_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        save_interval=50_000,
        fsdp_devices=8,
        action_horizon=50,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
