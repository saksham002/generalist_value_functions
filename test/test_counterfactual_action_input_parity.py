import dataclasses
import pathlib
import re
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_datasets as tfds

from openpi.robocoin_utils.utils import extract_embodiment
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.robocoin_utils.utils import RLDS_TO_STANDARD_CAMERA_MAP
import openpi.transforms as _transforms


SEED = 86
CONFIG_NAME = "robocoin_bimanual_pi05_rlds"
NUM_BATCHES_TO_SCAN = 2500
BATCH_SIZE = 4
DEBUG_IMAGE_DIR = pathlib.Path("test/images")


def _make_train_config() -> _config.TrainConfig:
    config = _config.get_config(CONFIG_NAME)
    return dataclasses.replace(config, batch_size = BATCH_SIZE, seed = SEED, num_workers = 0)


def _action_horizon(config: _config.TrainConfig) -> int:
    action_horizon = config.action_horizon
    if action_horizon is None:
        action_horizon = config.model.action_horizon
    if action_horizon is None:
        raise ValueError("Action horizon must be set on either TrainConfig or the model config.")
    return action_horizon


def _index_sample(tree, index: int):
    if isinstance(tree, dict):
        return {key: _index_sample(value, index) for key, value in tree.items()}
    return np.asarray(tree)[index]


def _normalize_repo_id(repo_id) -> str:
    repo_id = np.asarray(repo_id).item()
    if isinstance(repo_id, bytes):
        repo_id = repo_id.decode("utf-8")
    return repo_id


def _get_rlds_episode_index(episode: dict) -> int:
    return int(episode["steps"][0]["episode_index"])


def _sample_key(sample) -> tuple[str, int, int]:
    return (
        _normalize_repo_id(sample["repo_id"]),
        int(np.asarray(sample["episode_index"]).item()),
        int(np.asarray(sample["_frame_index"]).item()),
    )


def _build_policy_prompt(subtask_texts: list[str], first_null_index: int) -> str | None:
    valid_subtask_texts = []
    for text in subtask_texts[:first_null_index]:
        stripped = text.rstrip(". ").strip()
        lowered = stripped.lower()
        if lowered in {"static", "abnormal"}:
            continue
        valid_subtask_texts.append(stripped)
    if not valid_subtask_texts:
        return None
    return ", ".join(valid_subtask_texts)


def _assert_equal(lhs, rhs, path: str) -> None:
    if isinstance(lhs, dict):
        assert isinstance(rhs, dict), f"{path}: expected dict, got {type(rhs)}"
        assert set(lhs.keys()) == set(rhs.keys()), f"{path}: key mismatch {sorted(lhs.keys())} != {sorted(rhs.keys())}"
        for key in sorted(lhs.keys()):
            _assert_equal(lhs[key], rhs[key], f"{path}.{key}")
        return

    lhs_array = np.asarray(lhs)
    rhs_array = np.asarray(rhs)
    assert lhs_array.shape == rhs_array.shape, f"{path}: shape mismatch {lhs_array.shape} != {rhs_array.shape}"
    assert lhs_array.dtype == rhs_array.dtype, f"{path}: dtype mismatch {lhs_array.dtype} != {rhs_array.dtype}"
    if path.endswith(".image.base_0_rgb") and not np.array_equal(lhs_array, rhs_array):
        _write_base_image_debug(lhs_array, rhs_array, path)
    np.testing.assert_array_equal(lhs_array, rhs_array, err_msg = path)


def _write_base_image_debug(training_image: np.ndarray, counterfactual_image: np.ndarray, path: str) -> None:
    import matplotlib.pyplot as plt

    DEBUG_IMAGE_DIR.mkdir(parents = True, exist_ok = True)
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", path)[:200] + ".png"
    output_path = DEBUG_IMAGE_DIR / filename

    figure, axes = plt.subplots(1, 2, figsize = (10, 5))
    axes[0].imshow(training_image)
    axes[0].set_title("training")
    axes[0].axis("off")
    axes[1].imshow(counterfactual_image)
    axes[1].set_title("counterfactual")
    axes[1].axis("off")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _collect_training_samples() -> dict[tuple[str, int, int], dict]:
    config = _make_train_config()
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_rlds_dataset(
        data_config,
        _action_horizon(config),
        config.batch_size,
        split = "train",
        shuffle = False,
    )
    dataset = _data_loader.transform_iterable_dataset(dataset, data_config, is_batched = True)

    samples: dict[tuple[str, int, int], dict] = {}
    for batch_index, batch in enumerate(dataset):
        if batch_index >= NUM_BATCHES_TO_SCAN:
            break
        batch_size = next(np.asarray(value).shape[0] for value in batch.values() if hasattr(value, "shape"))
        for sample_index in range(batch_size):
            sample = _index_sample(batch, sample_index)
            key = _sample_key(sample)
            if key in samples:
                raise ValueError(f"Duplicate training sample for key={key}")
            samples[key] = {
                "image": sample["image"],
                "state": sample["state"],
                "tokenized_prompt": sample["tokenized_prompt"],
                "tokenized_prompt_mask": sample["tokenized_prompt_mask"],
            }
    return samples


def _collect_counterfactual_samples() -> dict[tuple[str, int, int], dict]:
    config = _make_train_config()
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset_cfg = data_config.datasets[0]
    builder = tfds.builder(dataset_cfg.name, data_dir = data_config.rlds_data_dir, version = dataset_cfg.version)

    input_transform = _transforms.compose(
        [
            _transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles = data_config.use_quantile_norm),
            *(
                [_transforms.Clip(data_config.clip_normalized_bounds)]
                if data_config.clip_normalized_bounds is not None
                else []
            ),
            *data_config.model_transforms.inputs,
        ]
    )

    samples: dict[tuple[str, int, int], dict] = {}
    episodes_seen = 0
    batch_budget = NUM_BATCHES_TO_SCAN * BATCH_SIZE

    for raw_episode in builder.as_dataset(split = "train"):
        episode = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in raw_episode.items()}
        episode["steps"] = list(episode["steps"])

        repo_id = _normalize_repo_id(episode["episode_metadata"]["repo_id"])
        episode_index = _get_rlds_episode_index(episode)
        embodiment = extract_embodiment(repo_id)
        for frame_index, step in enumerate(episode["steps"]):
            first_null_index = int(step["first_null_index"])
            if first_null_index == 0:
                continue

            subtask_texts = []
            for subtask_index in range(1, 6):
                text = step[f"subtask_{subtask_index}"]
                if hasattr(text, "numpy"):
                    text = text.numpy()
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
                subtask_texts.append(text)

            prompt = _build_policy_prompt(subtask_texts, first_null_index)
            if prompt is None:
                continue

            state = step["observation/state"]
            if hasattr(state, "numpy"):
                state = state.numpy()
            state = np.asarray(state, dtype = np.float32)

            decoded_images = {}
            for cam_key in ("observation/image/cam_0", "observation/image/cam_1", "observation/image/cam_2"):
                image_bytes = step[cam_key]
                if hasattr(image_bytes, "numpy"):
                    image_bytes = image_bytes.numpy()
                rlds_cam_key = cam_key.split("/")[-1]
                standard_cam_key = RLDS_TO_STANDARD_CAMERA_MAP[rlds_cam_key]
                decoded_images[standard_cam_key] = np.asarray(
                    tf.io.decode_image(image_bytes, expand_animations = False, dtype = tf.uint8).numpy()
                )

            transformed = input_transform(
                {
                    "image": decoded_images,
                    "state": state,
                    "prompt": prompt,
                    "embodiment": embodiment,
                }
            )
            transformed = {key: value for key, value in transformed.items() if not isinstance(value, str)}

            key = repo_id, episode_index, frame_index
            if key in samples:
                raise ValueError(f"Duplicate counterfactual sample for key={key}")
            samples[key] = {
                "image": transformed["image"],
                "state": transformed["state"],
                "tokenized_prompt": transformed["tokenized_prompt"],
                "tokenized_prompt_mask": transformed["tokenized_prompt_mask"],
            }

            if len(samples) >= batch_budget:
                return samples

        episodes_seen += 1

    return samples


@pytest.mark.manual
def test_counterfactual_inputs_match_rlds_training_inputs():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    training_samples = _collect_training_samples()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    counterfactual_samples = _collect_counterfactual_samples()

    matched_keys = sorted(set(training_samples) & set(counterfactual_samples))
    assert matched_keys, "No overlapping (repo_id, episode_index, _frame_index) triples found across scanned samples."

    print(f"training collected samples: {len(training_samples)}")
    print(f"counterfactual collected samples: {len(counterfactual_samples)}")
    print(f"matched data points compared exactly: {len(matched_keys)}")

    for key in matched_keys:
        _assert_equal(training_samples[key], counterfactual_samples[key], path = f"sample[{key}]")
