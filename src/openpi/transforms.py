from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools
from scipy.spatial.transform import Rotation

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        norm_stats = self._select_norm_stats(data)
        return apply_tree(
            data,
            norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _select_norm_stats(self, data: DataDict) -> at.PyTree[NormStats]:
        if not self.norm_stats:
            return self.norm_stats

        first_value = next(iter(self.norm_stats.values()))
        if isinstance(first_value, NormStats):
            return self.norm_stats

        if "embodiment" not in data:
            raise ValueError("Embodiment-keyed normalization requires 'embodiment' in the sample.")

        embodiment = data["embodiment"]
        if isinstance(embodiment, np.ndarray):
            embodiment = embodiment.item()
        if isinstance(embodiment, bytes):
            embodiment = embodiment.decode("utf-8")
        if embodiment not in self.norm_stats:
            raise ValueError(
                f"Missing normalization stats for embodiment '{embodiment}'. "
                f"Available embodiments: {sorted(self.norm_stats.keys())}"
            )
        return self.norm_stats[embodiment]

    def _normalize(self, x, stats: NormStats):
        return (x - stats.mean) / (stats.std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Clip(DataTransformFn):
    bounds: Mapping[str, tuple[float, float]]

    def __call__(self, data: DataDict) -> DataDict:
        for key, (low, high) in self.bounds.items():
            if key in data:
                data[key] = np.clip(data[key], low, high)
        return data


@dataclasses.dataclass(frozen=True)
class ReplaceMaskedActions(DataTransformFn):
    use_quantile_norm: bool = False
    rng: np.random.Generator = dataclasses.field(
        default_factory = lambda: np.random.default_rng(seed = 86),
        compare = False,
        repr = False,
    )

    def __call__(self, data: DataDict) -> DataDict:
        fps = int(np.asarray(data["fps"]).item())
        noise_scale = 0.002 if self.use_quantile_norm else 0.005

        for actions_key, mask_key in (
            ("actions", "action_mask"),
            ("next_actions", "next_action_mask"),
            ("counterfactual_actions", "action_mask"),
            ("counterfactual_next_actions", "next_action_mask"),
        ):
            if actions_key not in data:
                continue

            actions = np.asarray(data[actions_key]).copy()
            action_mask = np.asarray(data[mask_key], dtype = np.bool_).copy()
            if actions.ndim not in (2, 3):
                raise ValueError(f"{actions_key} must have rank 2 or 3, got shape {actions.shape}")

            last_valid_idx = max(int(action_mask.sum()) - 1, 0)

            if actions.ndim == 3:
                # [num_samples, action_horizon, action_dim] — broadcast mask across samples
                # Use the same noise across all samples so replacement is consistent
                last_valid_action = actions[:, last_valid_idx]
                shared_noise = (noise_scale * self.rng.standard_normal(actions.shape[1:])).astype(actions.dtype)
                replacement = last_valid_action[:, None, :] + shared_noise[None, :, :]
                mask_broadcast = action_mask[None, :, None]
                actions = np.where(mask_broadcast, actions, replacement)
            else:
                last_valid_action = actions[last_valid_idx]
                noise = (noise_scale * self.rng.standard_normal(actions.shape)).astype(actions.dtype)
                replacement = last_valid_action[None, :] + noise
                actions[~action_mask] = replacement[~action_mask]

            data[actions_key] = actions

        for mask_key in ("action_mask", "next_action_mask"):
            if mask_key not in data:
                continue
            action_mask = np.asarray(data[mask_key], dtype = np.bool_).copy()
            action_mask[:] = True
            if fps == 30:
                action_horizon = action_mask.shape[0]
                action_mask[3 * action_horizon // 5 :] = False
            data[mask_key] = action_mask

        return data


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        norm_stats = self._select_norm_stats(data)
        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _select_norm_stats(self, data: DataDict) -> at.PyTree[NormStats]:
        if not self.norm_stats:
            return self.norm_stats

        first_value = next(iter(self.norm_stats.values()))
        if isinstance(first_value, NormStats):
            return self.norm_stats

        if "embodiment" not in data:
            raise ValueError("Embodiment-keyed unnormalization requires 'embodiment' in the sample.")

        embodiment = data["embodiment"]
        if isinstance(embodiment, np.ndarray):
            embodiment = embodiment.item()
        if isinstance(embodiment, bytes):
            embodiment = embodiment.decode("utf-8")
        if embodiment not in self.norm_stats:
            raise ValueError(
                f"Missing normalization stats for embodiment '{embodiment}'. "
                f"Available embodiments: {sorted(self.norm_stats.keys())}"
            )
        return self.norm_stats[embodiment]

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        # Direct stretch matches dexterous_hang_config.py:_decode_and_reencode_jpeg, which
        # writes 224x224 training frames via tf.image.resize without aspect preservation.
        # Using resize_with_pad here would letterbox the eval frames and the model has
        # never seen black bars on top/bottom.
        data["image"] = {k: image_tools.resize_stretch(v, self.height, self.width) for k, v in data["image"].items()}
        if "next_image" in data:
            data["next_image"] = {
                k: image_tools.resize_stretch(v, self.height, self.width) for k, v in data["next_image"].items()
            }
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space.

    Position dims (mask=True) become `action - state`. Orientation blocks
    starting at each index in `rpy_index_start` (treated as three extrinsic
    xyz euler angles) become the euler angles of `R_action @ R_state.inv()`.
    Dims outside both the mask and the rpy blocks stay absolute.
    """

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None
    # Starting indices of 3-wide extrinsic-xyz euler blocks to be composed as relative rotations.
    rpy_index_start: Sequence[int] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if self.mask is None and self.rpy_index_start is None:
            return data

        if self.mask is not None:
            mask = np.asarray(self.mask).copy()
            # Zero the mask over rpy slots so they are not double-processed by the subtraction below.
            if self.rpy_index_start is not None:
                for s in self.rpy_index_start:
                    mask[s : s + 3] = False
            dims = mask.shape[-1]
        else:
            mask = None
            dims = 0

        for state_key, action_key in (("state", "actions"), ("next_state", "next_actions")):
            if action_key not in data or state_key not in data:
                continue
            state = data[state_key]
            # Force a copy so in-place updates below do not mutate the caller's array.
            actions = np.array(data[action_key])
            if mask is not None:
                actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis = -2)
            if self.rpy_index_start is not None:
                _apply_rpy_delta(state, actions, self.rpy_index_start)
            data[action_key] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space.

    Inverse of `DeltaActions`: position dims (mask=True) add state back,
    and rpy blocks are recomposed via `R_delta @ R_state` before being
    written back as extrinsic xyz euler angles.
    """

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None
    # Starting indices of 3-wide extrinsic-xyz euler blocks that were composed as relative rotations.
    rpy_index_start: Sequence[int] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or (self.mask is None and self.rpy_index_start is None):
            return data

        # Force a copy so in-place updates below do not mutate the caller's array.
        state, actions = data["state"], np.array(data["actions"])
        if self.mask is not None:
            mask = np.asarray(self.mask).copy()
            if self.rpy_index_start is not None:
                for s in self.rpy_index_start:
                    mask[s : s + 3] = False
            dims = mask.shape[-1]
            actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis = -2)
        if self.rpy_index_start is not None:
            _apply_rpy_absolute(state, actions, self.rpy_index_start)
        data["actions"] = actions

        return data


def _apply_rpy_delta(state, actions, rpy_index_start: Sequence[int]) -> None:
    """In-place: write extrinsic-xyz euler angles of R_action @ R_state.inv() into each rpy block."""
    state = np.asarray(state)
    for s in rpy_index_start:
        state_rpy = state[..., s : s + 3]
        action_rpy = actions[..., s : s + 3]
        # Broadcast state over the chunk axis so each action in the chunk composes with the same state.
        state_rpy_b = np.broadcast_to(state_rpy[..., None, :], action_rpy.shape)
        leading = action_rpy.shape[:-1]
        r_state = Rotation.from_euler("xyz", state_rpy_b.reshape(-1, 3))
        r_action = Rotation.from_euler("xyz", action_rpy.reshape(-1, 3))
        r_delta = r_action * r_state.inv()
        actions[..., s : s + 3] = r_delta.as_euler("xyz").reshape(*leading, 3)


def _apply_rpy_absolute(state, actions, rpy_index_start: Sequence[int]) -> None:
    """In-place: write extrinsic-xyz euler angles of R_delta @ R_state into each rpy block."""
    state = np.asarray(state)
    for s in rpy_index_start:
        state_rpy = state[..., s : s + 3]
        action_rpy = actions[..., s : s + 3]
        state_rpy_b = np.broadcast_to(state_rpy[..., None, :], action_rpy.shape)
        leading = action_rpy.shape[:-1]
        r_state = Rotation.from_euler("xyz", state_rpy_b.reshape(-1, 3))
        r_delta = Rotation.from_euler("xyz", action_rpy.reshape(-1, 3))
        r_abs = r_delta * r_state
        actions[..., s : s + 3] = r_abs.as_euler("xyz").reshape(*leading, 3)


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer | _tokenizer.Gemma3Tokenizer | _tokenizer.Gemma4Tokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeRoboCoinSubtaskPrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer | _tokenizer.Gemma3Tokenizer | _tokenizer.Gemma4Tokenizer
    prefix_text: str | None = None
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        prefix = data.pop("prompt", None)
        suffix = data.pop("subtask_text", None)
        if prefix is None or suffix is None:
            raise ValueError("Both prompt and subtask_text are required")

        if not isinstance(prefix, str):
            prefix = prefix.item()
        if not isinstance(suffix, str):
            suffix = suffix.item()

        if self.prefix_text is not None and prefix != self.prefix_text:
            raise ValueError(f"Expected prefix {self.prefix_text!r}, got {prefix!r}")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        tokens, token_masks, subtask_start_index, subtask_end_index = _tokenize_robocoin_subtask_prompt(
            self.tokenizer,
            prefix,
            suffix,
            state = state,
        )
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_masks,
            "subtask_start_index": np.int32(subtask_start_index),
            "subtask_end_index": np.int32(subtask_end_index),
        }


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension.

    When ``action_dim_mask`` is provided, source values are scattered into the
    True positions of the mask in order (i.e. source[..., i] lands at the
    i-th True index in mask). Number of source dims must equal the number of
    True entries.

    Otherwise, when ``action_dim_offset > 0``, values are placed at
    [offset : offset + d].
    """

    model_action_dim: int
    action_dim_offset: int = 0
    pad_state: bool = True
    action_dim_mask: tuple[bool, ...] | None = None

    def __call__(self, data: DataDict) -> DataDict:
        if self.action_dim_mask is not None:
            true_indices = tuple(i for i, m in enumerate(self.action_dim_mask) if m)
            if self.pad_state:
                data["state"] = _scatter_to_mask(data["state"], self.model_action_dim, true_indices)
            if "actions" in data:
                data["actions"] = _scatter_to_mask(data["actions"], self.model_action_dim, true_indices)
        elif self.action_dim_offset > 0:
            if self.pad_state:
                data["state"] = _insert_at_offset(data["state"], self.model_action_dim, self.action_dim_offset)
            if "actions" in data:
                data["actions"] = _insert_at_offset(data["actions"], self.model_action_dim, self.action_dim_offset)
        else:
            if self.pad_state:
                data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis = -1)
            if "actions" in data:
                data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis = -1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def _insert_at_offset(x: np.ndarray, target_dim: int, offset: int) -> np.ndarray:
    """Place x's last-axis values at [offset : offset + d] in a zero array of size target_dim."""
    real_dim = x.shape[-1]
    assert offset + real_dim <= target_dim, f"offset({offset}) + dim({real_dim}) > target({target_dim})"
    out_shape = x.shape[:-1] + (target_dim,)
    out = np.zeros(out_shape, dtype = x.dtype)
    out[..., offset : offset + real_dim] = x
    return out


def _scatter_to_mask(x: np.ndarray, target_dim: int, true_indices: tuple[int, ...]) -> np.ndarray:
    """Scatter x's last-axis values into true_indices of a zero array of size target_dim.

    Source dim i lands at target index true_indices[i]. Asserts source dim count
    equals len(true_indices).
    """
    real_dim = x.shape[-1]
    assert real_dim == len(true_indices), (
        f"source dim ({real_dim}) != number of True positions in mask ({len(true_indices)})"
    )
    assert all(0 <= idx < target_dim for idx in true_indices), (
        f"true_indices out of range for target_dim={target_dim}: {true_indices}"
    )
    out_shape = x.shape[:-1] + (target_dim,)
    out = np.zeros(out_shape, dtype = x.dtype)
    out[..., list(true_indices)] = x
    return out


def _clean_prompt_text(prompt: str) -> str:
    return prompt.strip().replace("_", " ").replace("\n", " ")


def _tokenize_robocoin_subtask_prompt(
    tokenizer: _tokenizer.PaligemmaTokenizer | _tokenizer.Gemma3Tokenizer | _tokenizer.Gemma4Tokenizer,
    prefix: str,
    suffix: str,
    *,
    state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if state is not None:
        raise NotImplementedError("TokenizeRoboCoinSubtaskPrompt does not support discrete state input.")

    cleaned_prefix = _clean_prompt_text(prefix)
    cleaned_suffix = _clean_prompt_text(suffix)
    prefix_with_separator = f"{cleaned_prefix} " if cleaned_prefix else cleaned_prefix

    # Tokenize prefix and suffix separately so the split index is correct by construction.
    # SentencePiece is non-compositional across boundaries — joining the strings before
    # encoding can shift tokens at the boundary and invalidate len(prefix_tokens) as the split.
    add_bos = getattr(tokenizer, "_use_bos", True)
    prefix_tokens = tokenizer._tokenizer.encode(prefix_with_separator, add_bos = add_bos)
    suffix_tokens = tokenizer._tokenizer.encode(cleaned_suffix, add_bos = False)
    newline_tokens = tokenizer._tokenizer.encode("\n")
    raw_tokens = prefix_tokens + suffix_tokens + newline_tokens
    subtask_start_index = len(prefix_tokens)
    subtask_end_index = subtask_start_index + len(suffix_tokens) - 1

    image_tokenizer = isinstance(tokenizer, (_tokenizer.Gemma3Tokenizer, _tokenizer.Gemma4Tokenizer))
    if image_tokenizer and tokenizer._num_images > 0:
        soi_markers = [tokenizer.START_OF_IMAGE_ID] * tokenizer._num_images
        raw_tokens = [raw_tokens[0]] + soi_markers + raw_tokens[1:]
        subtask_start_index += tokenizer._num_images
        subtask_end_index += tokenizer._num_images

    max_len = tokenizer._max_len + (tokenizer._num_images if image_tokenizer else 0)
    tokens_len = len(raw_tokens)
    if tokens_len < max_len:
        padding = [False] * (max_len - tokens_len)
        token_mask = [True] * tokens_len + padding
        raw_tokens = raw_tokens + padding
    else:
        raw_tokens = raw_tokens[:max_len]
        token_mask = [True] * max_len
        subtask_start_index = min(subtask_start_index, max_len)
        subtask_end_index = min(subtask_end_index, max_len - 1)

    return np.asarray(raw_tokens), np.asarray(token_mask), subtask_start_index, subtask_end_index


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
