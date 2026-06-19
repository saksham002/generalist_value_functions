"""RLDS data loader for LeRobot-built real-world datasets (e.g. ``realworld_xarm_packing``).

Trimmed sibling of ``Hdf5RldsDataset`` for datasets exported by the LeRobot RLDS
builder. Unlike the HDF5 schema, here ``observation/state`` and ``action`` are
already 14D EEF (``pos3 + euler3 + gripper`` per arm, no separate
``eef_sim_pose_*`` fields to reconstruct), the subtask annotation is a single
per-step ``subtask`` string, and every demo is an expert/complete trajectory
(no ``reward`` field, no ``is_partial`` heuristic). fps is 60 and
``action_chunk_size`` is 30 or 60.

Only the behavior-cloning path is supported (``critic_mode=False``): the base
``_prepare_trajectory`` skips the RL next-step / reward machinery, so this class
overrides just ``trajectory_transforms`` / ``frame_transforms`` / ``frame_filter``.
"""

from collections.abc import Sequence
import logging
from typing import Any, Literal

import openpi.training.rlds_dataset as rlds_dataset

PromptMode = Literal["subtask", "task_description", "task_description_predict_current_subtask"]


class LeRobotRldsDataset(rlds_dataset.BaseRldsDataset):
    """LeRobot-sourced RLDS dataset loader with single-subtask, expert-only semantics."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[rlds_dataset.RLDSDataset],
        *,
        split: str = "train",
        shuffle: bool = True,
        shuffle_seed: int = 86,
        action_chunk_size: int = 30,
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
        critic_mode: bool = False,
        discount: float = 0.99,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        filter_n: int | None = None,
        prompt_mode: PromptMode = "subtask",
        image_size: tuple[int, int] | None = None,
        include_images: bool = True,
        decode_images: bool = True,
        return_trajectories: bool = False,
        max_trajectories: int | None = None,
        max_num_demos: int | None = None,
    ):
        if action_chunk_size not in (30, 60):
            raise ValueError(f"action_chunk_size must be 30 or 60, got {action_chunk_size}")
        if filter_n is not None and filter_n % 2 != 0:
            raise ValueError(f"filter_n must be a multiple of 2, got {filter_n}")
        if critic_mode:
            raise ValueError("LeRobotRldsDataset only supports critic_mode=False (behavior cloning).")

        self._split = split
        self._filter_n = filter_n
        self._prompt_mode = prompt_mode
        logging.info(f"LeRobotRldsDataset: filter_n={filter_n}, prompt_mode={prompt_mode}")

        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            datasets=datasets,
            split=split,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            action_chunk_size=action_chunk_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_parallel_reads=num_parallel_reads,
            num_parallel_calls=num_parallel_calls,
            critic_mode=critic_mode,
            discount=discount,
            reward_scale=reward_scale,
            reward_bias=reward_bias,
            image_obs_keys=("cam_0", "cam_1", "cam_2"),
            image_size=image_size,
            include_images=include_images,
            decode_images=decode_images,
            return_trajectories=return_trajectories,
            max_trajectories=max_trajectories,
            max_num_demos=max_num_demos,
        )

    def trajectory_transforms(self, traj: dict, dataset_cfg: rlds_dataset.RLDSDataset) -> dict:
        """Extract the per-trajectory fields consumed by ``frame_transforms`` / ``frame_filter``."""
        import tensorflow as tf

        del dataset_cfg

        actions = tf.cast(traj["action"], tf.float32)
        state = tf.cast(traj["observation/state"], tf.float32)
        tf.debugging.assert_equal(
            tf.shape(state)[-1],
            14,
            message="LeRobotRldsDataset requires 14D observation/state",
        )
        tf.debugging.assert_equal(
            tf.shape(actions)[-1],
            14,
            message="LeRobotRldsDataset requires 14D action",
        )

        fps = traj["traj_metadata"]["episode_metadata"]["fps"]
        tf.debugging.assert_equal(
            tf.cast(fps[0], tf.int32),
            tf.constant(60, dtype=tf.int32),
            message="LeRobotRldsDataset requires fps=60 source data.",
        )

        observation = {"state": state}
        if self._include_images:
            observation["cam_0"] = traj["observation/image/cam_0"]
            observation["cam_1"] = traj["observation/image/cam_1"]
            observation["cam_2"] = traj["observation/image/cam_2"]

        # task_description has no subtasks: count down to episode end instead.
        if self._prompt_mode == "task_description":
            steps_to_subtask_end = tf.cast(traj["_len"] - 1 - traj["_frame_index"], tf.int32)
        else:
            steps_to_subtask_end = tf.cast(traj["steps_to_subtask_end"], tf.int32)

        return {
            "actions": actions,
            "observation": observation,
            "subtask": traj["subtask"],
            "task_description": traj["traj_metadata"]["episode_metadata"]["task_description"],
            "steps_to_subtask_end": steps_to_subtask_end,
        }

    def _restructure_images(self, frame: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Move camera images from the observation dict into the standard image/image_mask format."""
        import tensorflow as tf

        images = {}
        masks = {}
        image_name_map = {
            "cam_0": "base_0_rgb",
            "cam_1": "left_wrist_0_rgb",
            "cam_2": "right_wrist_0_rgb",
        }
        for key, value in frame["observation"].items():
            if key == "state":
                continue
            image_key = image_name_map[key]
            images[image_key] = value
            masks[image_key] = tf.constant(True)
        return images, masks

    def frame_transforms(self, frame: dict) -> dict:
        """Decode images, set the prompt, restructure to standard openpi keys."""
        import tensorflow as tf

        if self._prompt_mode == "task_description":
            frame["prompt"] = frame["task_description"]
        elif self._prompt_mode == "task_description_predict_current_subtask":
            frame["prompt"] = frame["task_description"]
            frame["subtask_text"] = frame["subtask"]
        else:
            frame["prompt"] = frame["subtask"]

        frame = super().frame_transforms(frame)

        images, image_masks = self._restructure_images(frame)
        frame["state"] = tf.cast(frame["observation"]["state"], tf.float32)
        del frame["observation"]

        frame["actions"] = tf.cast(frame["actions"], tf.float32)
        if images:
            frame["image"] = images
            frame["image_mask"] = image_masks

        return frame

    def frame_filter(self, frame: dict) -> bool:
        """Drop frames within ``filter_n`` steps of the (sub)task end."""
        import tensorflow as tf

        keep = tf.constant(True)
        if self._filter_n is not None:
            keep = tf.logical_and(keep, frame["steps_to_subtask_end"] >= self._filter_n)
        return keep
