"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

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
from openpi.policy_extraction import objectives as _policy_extraction
import openpi.shared.download as _download

try:
    import openpi.shared.legacy_d4rl_utils as legacy_d4rl_utils
except Exception:
    legacy_d4rl_utils = None  # type: ignore
import openpi.shared.minari_utils as minari_utils
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
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
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()

    # RL training mode options (for value function training)
    rl_mode: bool = False  # If True, use value function training pipeline
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

    # RoboCOIN-specific data loader config (if set, uses DLIMP-based loader)
    # This is set by RoboCOINDataConfig.create() and detected by create_data_loader()
    robocoin_data_config: Any | None = None


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
                        _transforms.PadStatesAndActions(model_config.action_dim),
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
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
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

    When rl_mode=True, the data loader will:
    1. Load transitions as SARSA tuples: (s, a, r, s', a')
    2. Compute discounted Monte-Carlo returns for each trajectory
    3. Include 'mc_return' in the data dict for value function training
    """

    # Action dimension for the D4RL environment. If None, will be inferred from model_config.
    action_dim: int | None = None
    # Default task name (environment name) if not provided in data
    default_task: str | None = None

    # RL training mode options
    rl_mode: bool = False  # If True, load SARSA tuples + MC returns
    discount: float = 0.99  # Discount factor for MC return computation

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # No repack needed - D4RL LeRobot datasets already have 'state' and 'actions' keys
        repack_transform = _transforms.Group(inputs=[])

        if self.rl_mode:
            # RL mode: use value function transforms
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
            rl_mode=self.rl_mode,
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
            rl_mode=True,
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
            rl_mode=True,
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
class RoboCOINDataConfig(DataConfigFactory):
    """Data config for RoboCOIN TFDS dataset with images, text, and state.

    This config enables DLIMP-based data loading for the RoboCOIN dataset,
    which contains robot manipulation trajectories with:
    - Camera images (up to 3 views)
    - Proprioceptive state
    - Task descriptions (text prompts)
    - Actions
    - Rewards

    Uses the custom DLIMP data loader for efficient image-heavy data loading.

    Normalization:
    - use_quantile_norm=False: z-score using mean/std keys from norm_stats.json
    - use_quantile_norm=True: min-max using min/max keys from norm_stats.json
    """

    # Path to TFDS data directory
    tfds_data_dir: str = "/data/group_data/rl/saksham3/"
    # Dataset name and version
    dataset_name: str = "robocoin:1.0.0"
    # Maximum number of camera views
    max_cameras: int = 3
    # Target image size (H, W)
    image_size: tuple[int, int] = (224, 224)
    # Maximum state dimension (for padding)
    max_state_dim: int = 118
    # Maximum action dimension (for padding)
    max_action_dim: int = 54
    # Discount factor for MC return computation
    discount: float = 0.99
    # Reward transformation: r' = reward_scale * r + reward_bias
    reward_scale: float = 1.0
    reward_bias: float = 0.0
    # Local (per-host) shuffle buffer size for frame-level shuffling
    local_shuffle_buffer_size: int = 50000
    # TD-n parameter for temporal difference learning
    # - None: MC (Monte Carlo) learning - uses full episode return
    # - int: TD-n learning - bootstraps with value at t + td_n
    td_n: int | None = None

    # Whether to use end-effector position state instead of joint angles.
    # If True, constructs 14-D state as: [eef_sim_pose_state[:6], state[6], eef_sim_pose_state[6:12], state[13]]
    # where eef_sim_pose_state is 12-D (6 left EEF + 6 right EEF) and state[6], state[13] are grippers.
    # Requires "eef_sim_pose_state" key in norm_stats.json for normalization of EEF components.
    use_eef: bool = False

    # Path to norm_stats.json file (RoboCOIN-specific format)
    # Expected format: {"observation.state": {"mean": [...], "std": [...], "min": [...], "max": [...]}}
    # For use_eef=True, also requires: {"eef_sim_pose_state": {"mean": [...], "std": [...], ...}}
    # Supports both local paths and GCS paths (gs://...)
    norm_stats_path: str | None = "gs://saksham-euw4/robocoin/norm_stats/norm_stats.json"

    # Normalization method:
    # - False: z-score normalization using mean/std keys
    # - True: min-max normalization using min/max keys (mapped to q01/q99 for quantile transform)
    use_quantile_norm: bool = False

    # Number of actions in the action chunk
    action_horizon: int = 30

    # If set, filter out frames where the sampled subtask's steps_to_subtask_end < filter_n
    filter_n: int | None = None

    # If True, mask out 50fps samples (loss_mask = False for fps != 30)
    mask_50fps: bool = False

    # Override repo_id from parent - not used for RoboCOIN
    repo_id: str = "robocoin"

    def _load_robocoin_norm_stats(self) -> dict[str, _transforms.NormStats] | None:
        """Load normalization stats from RoboCOIN-specific JSON format.

        The JSON format uses nested keys like "observation.state" with:
        - mean, std: for z-score normalization (use_quantile_norm=False)
        - min, max: for min-max normalization (use_quantile_norm=True)

        Maps to NormStats:
        - mean/std -> mean/std (z-score)
        - min/max -> q01/q99 (quantile transform does min-max)

        Supports both local paths and GCS paths (gs://...).

        Raises:
            FileNotFoundError: If norm_stats_path is set but file doesn't exist.
            ValueError: If required keys are missing from the JSON.
        """
        if self.norm_stats_path is None:
            return None

        import json

        import numpy as np

        path_str = self.norm_stats_path

        # Use tf.io.gfile for GCS paths, standard file I/O otherwise
        if path_str.startswith("gs://"):
            import tensorflow as tf

            if not tf.io.gfile.exists(path_str):
                raise FileNotFoundError(
                    f"RoboCOIN norm_stats file not found at GCS path: {path_str}\n"
                    f"Please upload the norm_stats.json file or set norm_stats_path=None to skip normalization."
                )

            with tf.io.gfile.GFile(path_str, "r") as f:
                data = json.load(f)
        else:
            path = pathlib.Path(path_str)
            if not path.exists():
                raise FileNotFoundError(
                    f"RoboCOIN norm_stats file not found at: {path}\n"
                    f"Please create the norm_stats.json file or set norm_stats_path=None to skip normalization."
                )

            with open(path) as f:
                data = json.load(f)

        norm_stats = {}

        # Check required keys exist
        if "observation.state" not in data:
            raise ValueError(f"norm_stats.json requires 'observation.state' key, but found: {list(data.keys())}")

        state_stats = data["observation.state"]

        if self.use_eef:
            # EEF mode: construct combined norm stats from eef_sim_pose_state + observation.state grippers
            if "eef_sim_pose_state" not in data:
                raise ValueError(
                    f"use_eef=True requires 'eef_sim_pose_state' key in norm_stats.json, but found: {list(data.keys())}"
                )
            eef_stats = data["eef_sim_pose_state"]

            if self.use_quantile_norm:
                # Min-max normalization
                if "min" not in state_stats or "max" not in state_stats:
                    raise ValueError("use_quantile_norm=True requires 'min' and 'max' keys in observation.state")
                if "min" not in eef_stats or "max" not in eef_stats:
                    raise ValueError("use_quantile_norm=True requires 'min' and 'max' keys in eef_sim_pose_state")

                # Construct combined: [eef[:6], state[6], eef[6:12], state[13]]
                combined_min = np.concatenate(
                    [
                        np.array(eef_stats["min"])[:6],
                        np.array(state_stats["min"])[6:7],
                        np.array(eef_stats["min"])[6:12],
                        np.array(state_stats["min"])[13:14],
                    ]
                )
                combined_max = np.concatenate(
                    [
                        np.array(eef_stats["max"])[:6],
                        np.array(state_stats["max"])[6:7],
                        np.array(eef_stats["max"])[6:12],
                        np.array(state_stats["max"])[13:14],
                    ]
                )
                norm_stats["state"] = _transforms.NormStats(
                    mean=None,
                    std=None,
                    q01=combined_min,
                    q99=combined_max,
                )
            else:
                # Z-score normalization
                if "mean" not in state_stats or "std" not in state_stats:
                    raise ValueError("use_quantile_norm=False requires 'mean' and 'std' keys in observation.state")
                if "mean" not in eef_stats or "std" not in eef_stats:
                    raise ValueError("use_quantile_norm=False requires 'mean' and 'std' keys in eef_sim_pose_state")

                # Construct combined: [eef[:6], state[6], eef[6:12], state[13]]
                combined_mean = np.concatenate(
                    [
                        np.array(eef_stats["mean"])[:6],
                        np.array(state_stats["mean"])[6:7],
                        np.array(eef_stats["mean"])[6:12],
                        np.array(state_stats["mean"])[13:14],
                    ]
                )
                combined_std = np.concatenate(
                    [
                        np.array(eef_stats["std"])[:6],
                        np.array(state_stats["std"])[6:7],
                        np.array(eef_stats["std"])[6:12],
                        np.array(state_stats["std"])[13:14],
                    ]
                )
                norm_stats["state"] = _transforms.NormStats(
                    mean=combined_mean,
                    std=combined_std,
                    q01=None,
                    q99=None,
                )
        # Joint angle mode: use observation.state directly
        elif self.use_quantile_norm:
            # Min-max normalization: use min/max mapped to q01/q99
            if "min" not in state_stats or "max" not in state_stats:
                raise ValueError(
                    f"use_quantile_norm=True requires 'min' and 'max' keys in norm_stats, "
                    f"but found: {list(state_stats.keys())}"
                )
            norm_stats["state"] = _transforms.NormStats(
                mean=None,  # Not used for quantile
                std=None,  # Not used for quantile
                q01=np.array(state_stats["min"]),
                q99=np.array(state_stats["max"]),
            )
        else:
            # Z-score normalization: use mean/std
            if "mean" not in state_stats or "std" not in state_stats:
                raise ValueError(
                    f"use_quantile_norm=False requires 'mean' and 'std' keys in norm_stats, "
                    f"but found: {list(state_stats.keys())}"
                )
            norm_stats["state"] = _transforms.NormStats(
                mean=np.array(state_stats["mean"]),
                std=np.array(state_stats["std"]),
                q01=None,
                q99=None,
            )

        # Add action_chunk normalization stats (for Q(s,a) training)
        if "action_diff" in data:
            action_stats = data["action_diff"]

            if self.use_eef:
                # EEF mode: construct combined action stats from eef_sim_pose_action_diff + action_diff grippers
                eef_action_stats = data["eef_sim_pose_action_diff"]

                if self.use_quantile_norm:
                    # Min-max normalization
                    combined_action_min = np.concatenate(
                        [
                            np.array(eef_action_stats["min"])[:6],
                            np.array(action_stats["min"])[6:7],
                            np.array(eef_action_stats["min"])[6:12],
                            np.array(action_stats["min"])[13:14],
                        ]
                    )
                    combined_action_max = np.concatenate(
                        [
                            np.array(eef_action_stats["max"])[:6],
                            np.array(action_stats["max"])[6:7],
                            np.array(eef_action_stats["max"])[6:12],
                            np.array(action_stats["max"])[13:14],
                        ]
                    )
                    norm_stats["actions"] = _transforms.NormStats(
                        mean=None,
                        std=None,
                        q01=combined_action_min,
                        q99=combined_action_max,
                    )
                else:
                    # Z-score normalization
                    combined_action_mean = np.concatenate(
                        [
                            np.array(eef_action_stats["mean"])[:6],
                            np.array(action_stats["mean"])[6:7],
                            np.array(eef_action_stats["mean"])[6:12],
                            np.array(action_stats["mean"])[13:14],
                        ]
                    )
                    combined_action_std = np.concatenate(
                        [
                            np.array(eef_action_stats["std"])[:6],
                            np.array(action_stats["std"])[6:7],
                            np.array(eef_action_stats["std"])[6:12],
                            np.array(action_stats["std"])[13:14],
                        ]
                    )
                    norm_stats["actions"] = _transforms.NormStats(
                        mean=combined_action_mean,
                        std=combined_action_std,
                        q01=None,
                        q99=None,
                    )
            # Joint angle mode: use action stats directly
            elif self.use_quantile_norm:
                norm_stats["actions"] = _transforms.NormStats(
                    mean=None,
                    std=None,
                    q01=np.array(action_stats["min"]),
                    q99=np.array(action_stats["max"]),
                )
            else:
                norm_stats["actions"] = _transforms.NormStats(
                    mean=np.array(action_stats["mean"]),
                    std=np.array(action_stats["std"]),
                    q01=None,
                    q99=None,
                )

        # Add next_state and next_actions with same normalization as their current counterparts
        if "state" in norm_stats:
            norm_stats["next_state"] = norm_stats["state"]
        if "actions" in norm_stats:
            norm_stats["next_actions"] = norm_stats["actions"]

        logging.info(f"Loaded RoboCOIN norm_stats from {path_str}, keys: {list(norm_stats.keys())}")
        return norm_stats if norm_stats else None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Value function transforms for RL training
        data_transforms = _transforms.Group(
            inputs=[_value_transforms.ValueFunctionInputs()],
            outputs=[],
        )
        model_transforms = _transforms.Group(inputs=[], outputs=[])

        # Use dataset name as asset_id
        asset_id = self.dataset_name.replace(":", "_").replace("/", "_")

        # Load RoboCOIN-specific norm stats
        norm_stats = self._load_robocoin_norm_stats()

        # Create the RoboCOIN data loader config
        robocoin_loader_config = self.get_data_loader_config()

        # Store RoboCOIN-specific config in a way the data loader can access
        # The training script will detect robocoin_data_config and use the custom loader
        return DataConfig(
            repo_id=None,  # Not using LeRobot
            asset_id=asset_id,
            norm_stats=norm_stats,
            repack_transforms=_transforms.Group(inputs=[]),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=self.use_quantile_norm,
            rl_mode=True,
            discount=self.discount,
            reward_scale=self.reward_scale,
            reward_bias=self.reward_bias,
            # Store RoboCOIN loader config for detection by create_data_loader
            robocoin_data_config=robocoin_loader_config,
        )

    def get_data_loader_config(self):
        """Return the RoboCOINDataLoaderConfig for the custom data loader."""
        from openpi.training.robocoin_data_loader import RoboCOINDataLoaderConfig

        return RoboCOINDataLoaderConfig(
            data_dir=self.tfds_data_dir,
            dataset_name=self.dataset_name,
            max_cameras=self.max_cameras,
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            image_size=self.image_size,
            discount=self.discount,
            reward_scale=self.reward_scale,
            reward_bias=self.reward_bias,
            local_shuffle_buffer_size=self.local_shuffle_buffer_size,
            td_n=self.td_n,
            use_eef=self.use_eef,
            action_horizon=self.action_horizon,
            filter_n=self.filter_n,
            mask_50fps=self.mask_50fps,
        )


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
    ema_decay: float | None = 0.99

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

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

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
    #
    # RoboCOIN PaliGemma V(s) value function configs.
    #
    TrainConfig(
        name="debug_robocoin_paligemma",
        model=_value_function.MCValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,  # Proprioceptive state dimension for RoboCOIN
                num_cameras=3,  # cam_0, cam_1, cam_2
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,  # Max tokens for subtask text
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="gs://saksham-euw4/robocoin_bimanual",
            dataset_name="robocoin:1.0.0",
            discount=0.999,
            local_shuffle_buffer_size=50000,
            norm_stats_path="gs://saksham-euw4/robocoin_bimanual/norm_stats/norm_stats.json"
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=120_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=120_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,  # DLIMP handles its own parallelism
        log_interval=100,
        plot_interval=200_000,
        save_interval=200_000,
        fsdp_devices=16,
        # wandb_enabled=False,
        validation_cache_dir="/nfs/aidm_nfs/saksham/robocoin/val_episodes_cache/",
    ),
    TrainConfig(
        name="robocoin_paligemma_v_mc",
        model=_value_function.MCValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,  # Proprioceptive state dimension for RoboCOIN
                num_cameras=3,  # cam_0, cam_1, cam_2
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,  # Max tokens for subtask text
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="/data/group_data/rl/saksham3/",
            dataset_name="robocoin:1.0.0",
            discount=0.99,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=30_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,  # DLIMP handles its own parallelism
        log_interval=100,
        plot_interval=5_000,
        save_interval=5_000,
        fsdp_devices=16,
        validation_cache_dir="/nfs/aidm_nfs/saksham/robocoin/val_episodes_cache_counterfactual/",
    ),
    TrainConfig(
        name="robocoin_paligemma_v_mc_use_eef",
        model=_value_function.MCValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,  # Proprioceptive state dimension for RoboCOIN
                num_cameras=3,  # cam_0, cam_1, cam_2
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,  # Max tokens for subtask text
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="/data/group_data/rl/saksham3/",
            dataset_name="robocoin:1.0.0",
            norm_stats_path="gs://saksham-euw4/robocoin/norm_stats/norm_stats.json",
            discount=0.99,
            use_eef=True,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=30_005,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,  # DLIMP handles its own parallelism
        log_interval=100,
        plot_interval=5_000,
        save_interval=5_000,
        fsdp_devices=16,
        validation_cache_dir="/nfs/aidm_nfs/saksham/robocoin/val_episodes_cache_counterfactual/",
    ),
    # RoboCOIN V(s) with mask_state=True (regression) - ablation: state masked out
    TrainConfig(
        name="robocoin_paligemma_v_mc_use_eef_no_state",
        model=_value_function.MCValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,
                mask_state=True,  # Ablation: mask out state token
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="/data/group_data/rl/saksham3/",
            dataset_name="robocoin:1.0.0",
            norm_stats_path="gs://saksham-euw4/robocoin/norm_stats/norm_stats.json",
            discount=0.99,
            use_eef=True,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=30_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=5_000,
        save_interval=5_000,
        fsdp_devices=16,
        validation_cache_dir="/nfs/aidm_nfs/saksham/robocoin/val_episodes_cache_counterfactual/",
    ),
    # RoboCOIN V(s) with mask_state=True (HL-Gauss) - ablation: state masked out
    TrainConfig(
        name="robocoin_paligemma_v_mc_use_eef_no_state_hl_gauss",
        model=_value_function.MCValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,
                mask_state=True,  # Ablation: mask out state token
            ),
            head_config=_heads.CategoricalHeadConfig(
                v_min=0.0,
                v_max=1.0,
                num_bins=51,
                sigma=0.015,
            ),
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="/data/group_data/rl/saksham3/",
            dataset_name="robocoin:1.0.0",
            norm_stats_path="gs://saksham-euw4/robocoin/norm_stats/norm_stats.json",
            discount=0.99,
            use_eef=True,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=30_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=5_000,
        save_interval=5_000,
        fsdp_devices=16,
        validation_cache_dir="/nfs/aidm_nfs/saksham/robocoin/val_episodes_cache_counterfactual/",
    ),
    # RoboCOIN Q(s,a) with MC regression - action-conditioned value function
    TrainConfig(
        name="robocoin_paligemma_q_mc", # use_eef is True 
        model=_value_function.MCValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,
                action_conditioned=True,
                action_dim=14,
                action_horizon=15,
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="/data/group_data/rl/saksham3/",
            dataset_name="robocoin:1.0.0",
            norm_stats_path="gs://saksham-euw4/robocoin/norm_stats/norm_stats.json",
            # norm_stats_path="/data/group_data/rl/saksham3/robocoin/norm_stats/norm_stats.json",
            discount=0.99,
            use_eef=True,
            action_horizon=15,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=30_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,  # DLIMP handles its own parallelism
        log_interval=100,
        plot_interval=5_000,
        save_interval=5_000,
        fsdp_devices=16,
        validation_cache_dir="/nfs/aidm_nfs/saksham/robocoin/val_episodes_cache_counterfactual/",
    ),
    # RoboCOIN Q(s,a) with MC regression and filtering - action-conditioned value function
    TrainConfig(
        name="robocoin_paligemma_filter_q_mc", # use_eef is True 
        model=_value_function.MCValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,
                action_conditioned=True,
                action_dim=14,
                action_horizon=15,
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="/data/group_data/rl/saksham3/",
            dataset_name="robocoin:1.0.0",
            norm_stats_path="gs://saksham-euw4/robocoin/norm_stats/norm_stats.json",
            # norm_stats_path="/data/group_data/rl/saksham3/robocoin/norm_stats/norm_stats.json",
            discount=0.99,
            use_eef=True,
            action_horizon=15,
            filter_n=15,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=30_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,  # DLIMP handles its own parallelism
        log_interval=100,
        plot_interval=5_000,
        save_interval=5_000,
        fsdp_devices=16,
        validation_cache_dir="/nfs/aidm_nfs/saksham/robocoin/val_episodes_cache_counterfactual/",
    ),
    # RoboCOIN MC value function with HL-Gauss (soft categorical) loss.
    TrainConfig(
        name="robocoin_paligemma_v_mc_hl_gauss",
        model=_value_function.MCValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,  # Proprioceptive state dimension for RoboCOIN
                num_cameras=3,
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,
            ),
            head_config=_heads.CategoricalHeadConfig(
                v_min=0.0,  # MC return = gamma^steps is in [0, 1]
                v_max=1.0,
                num_bins=51,  # Bin size is 0.02
                sigma=0.015,  # ratio of sigma to bin size is 0.75
            ),
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="/data/group_data/rl/saksham3/",
            dataset_name="robocoin:1.0.0",
            discount=0.99,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=30_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=5_000,
        save_interval=5_000,
        fsdp_devices=16,
        validation_cache_dir="/nfs/aidm_nfs/saksham/robocoin/val_episodes_cache_counterfactual/",
    ),
    # RoboCOIN V(s) with MC regression, EEF state, bimanual dataset.
    TrainConfig(
        name="robocoin_bimanual_paligemma_v_mc",
        model=_value_function.MCValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="gs://saksham-euw4/robocoin_bimanual",
            dataset_name="robocoin:1.0.0",
            norm_stats_path="gs://saksham-euw4/robocoin_bimanual/norm_stats/norm_stats.json",
            discount=0.999,
            use_eef=True,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=120_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=120_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=10_000,
        save_interval=10_000,
        fsdp_devices=16,
        num_val_trajectories=10,
        include_repos=("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning"),
        validation_cache_dir="/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache/",
    ),
    # RoboCOIN V(s) with MC regression, EEF state, bimanual dataset, 50fps masked out.
    TrainConfig(
        name="robocoin_bimanual_paligemma_v_mc_mask50fps",
        model=_value_function.MCValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,
            ),
            head_config=_heads.RegressionHeadConfig(),
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="gs://saksham-euw4/robocoin_bimanual",
            dataset_name="robocoin:1.0.0",
            norm_stats_path="gs://saksham-euw4/robocoin_bimanual/norm_stats/norm_stats.json",
            discount=0.999,
            use_eef=True,
            mask_50fps=True,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=120_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=120_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,
        log_interval=100,
        plot_interval=10_000,
        save_interval=10_000,
        fsdp_devices=16,
        num_val_trajectories=10,
        include_repos=("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning"),
        validation_cache_dir="/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache/",
    ),
    # RoboCOIN V(s) with MC one-hot cross-entropy loss, EEF state, bimanual dataset.
    TrainConfig(
        name="robocoin_bimanual_paligemma_v_mc_one_hot",
        model=_value_function.MCValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,
            ),
            head_config=_heads.CrossEntropyHeadConfig(
                v_min=0.0,
                v_max=1.0,
                num_bins=101,
            ),
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="gs://saksham-euw4/robocoin_bimanual",
            dataset_name="robocoin:1.0.0",
            norm_stats_path="gs://saksham-euw4/robocoin_bimanual/norm_stats/norm_stats.json",
            discount=0.999,
            use_eef=True,
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
        plot_interval=10_000,
        save_interval=10_000,
        fsdp_devices=16,
        num_val_trajectories=10,
        include_repos=("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning"),
        validation_cache_dir="/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache/",
    ),
    # RoboCOIN TD-15 value function config (regression loss).
    TrainConfig(
        name="robocoin_paligemma_v_td15",
        model=_value_function.SARSAValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,  # Proprioceptive state dimension for RoboCOIN
                num_cameras=3,  # cam_0, cam_1, cam_2
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,  # Max tokens for subtask text
            ),
            head_config=_heads.RegressionHeadConfig(),
            discount=0.99**15,  # Effective discount for 15-step return
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="/data/group_data/rl/saksham3/",
            dataset_name="robocoin:1.0.0",
            discount=0.99,
            td_n=15,  # TD-15: bootstrap with value at t + 15
            use_eef=True,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=30_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,  # DLIMP handles its own parallelism
        log_interval=100,
        plot_interval=5_000,
        fsdp_devices=16,
        validation_cache_dir="/nfs/aidm_nfs/saksham/robocoin/val_episodes_cache/",
    ),
    # RoboCOIN Q(s,a) SARSA with bimanual dataset. filter_n = td_n + 1 so that
    # terminal action_diff (a[t+1] - a[t] where t+1 is beyond subtask end) is not used.
    TrainConfig(
        name="robocoin_bimanual_paligemma_q_mc",
        model=_value_function.MCValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,
                num_cameras=3,
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,
                action_conditioned=True,
                action_dim=14,
                action_horizon=15,
            ),
            head_config=_heads.CategoricalHeadConfig(
                v_min=0.0,
                v_max=1.0,
                num_bins=101,
                sigma=0.0075,  # 0.75 * bin_width (bin_width = 1/100 = 0.01)
            ),
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="gs://saksham-euw4/robocoin_bimanual",
            dataset_name="robocoin:1.0.0",
            norm_stats_path="gs://saksham-euw4/robocoin_bimanual/norm_stats/norm_stats.json",
            discount=0.999,
            td_n=15,
            use_eef=True,
            action_horizon=15,
            filter_n=16,
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
        plot_interval=10_000,
        fsdp_devices=16,
        num_val_trajectories=24,
        include_repos=("RoboCOIN/Split_aloha_plate_storage", "RoboCOIN/Cobot_Magic_cut_banana", "RoboCOIN/R1_Lite_tableware_cleaning"),
        validation_cache_dir="/nfs/aidm_nfs/saksham3/robocoin/val_episodes_cache/",
    ),
    # RoboCOIN QC value function config (regression loss).
    TrainConfig(
        name="robocoin_paligemma_q_sarsa",
        model=_value_function.SARSAValueFunctionConfig(
            network_config=_paligemma_network.PaliGemmaNetworkConfig(
                state_dim=14,  # Proprioceptive state dimension for RoboCOIN
                num_cameras=3,  # cam_0, cam_1, cam_2
                image_size=(224, 224),
                freeze_backbone=False,
                max_token_len=48,  # Max tokens for subtask text
                action_conditioned=True,
                action_dim=14,
                action_horizon=15,
            ),
            head_config=_heads.RegressionHeadConfig(),
            discount=0.99**15,  # Effective discount for 15-step return
        ),
        data=RoboCOINDataConfig(
            tfds_data_dir="/data/group_data/rl/saksham3/",
            dataset_name="robocoin:1.0.0",
            discount=0.99,
            td_n=15,  # TD-15: bootstrap with value at t + 15
            use_eef=True,
            action_horizon=15,
        ),
        weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=30_000,
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-5,
            decay_steps=30_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(weight_decay=1e-6),
        num_workers=0,  # DLIMP handles its own parallelism
        log_interval=100,
        plot_interval=5_000,
        fsdp_devices=16,
        validation_cache_dir="/nfs/aidm_nfs/saksham/robocoin/val_episodes_cache/",
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
