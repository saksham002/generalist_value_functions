"""Online subtask detectors for real-world evaluation."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any

import numpy as np


def _last_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    return array[-1] if array.ndim > 1 else array


@dataclasses.dataclass
class DetectorState:
    """Base state shared by online subtask detectors."""

    current_subtask_index: int = 0
    sampled_steps: int = 0


class SubtaskDetector(abc.ABC):
    """Generic forward-only online subtask detector."""

    def __init__(self, subtasks: tuple[str, ...], *, sample_every_n_steps: int = 2):
        if not subtasks:
            raise ValueError("SubtaskDetector requires at least one subtask.")
        if sample_every_n_steps <= 0:
            raise ValueError(f"sample_every_n_steps must be positive, got {sample_every_n_steps}")
        self._subtasks = subtasks
        self._sample_every_n_steps = sample_every_n_steps
        self._state = self.initial_state()

    @property
    def subtasks(self) -> tuple[str, ...]:
        return self._subtasks

    @property
    def current_prompt(self) -> str:
        return self._subtasks[self._state.current_subtask_index]

    def reset(self) -> None:
        self._state = self.initial_state()

    def update(self, env_obs: dict, step_idx: int) -> str:
        if step_idx % self._sample_every_n_steps != 0:
            return self.current_prompt
        if self._state.current_subtask_index >= len(self._subtasks) - 1:
            return self.current_prompt
        self.observe_sample(env_obs, step_idx)
        self._advance_if_ready()
        return self.current_prompt

    def _advance_if_ready(self) -> None:
        while self._state.current_subtask_index < len(self._subtasks) - 1:
            next_index = self._state.current_subtask_index + 1
            if not self.should_advance_to(next_index):
                break
            self._state.current_subtask_index = next_index

    @abc.abstractmethod
    def initial_state(self) -> DetectorState:
        """Return the initial detector state."""

    @abc.abstractmethod
    def observe_sample(self, env_obs: dict, step_idx: int) -> None:
        """Update detector state from the sampled environment observation."""

    @abc.abstractmethod
    def should_advance_to(self, subtask_index: int) -> bool:
        """Return True when the detector is ready to advance to the given subtask."""


SHIRT_HANG_SUBTASKS = (
    "Grasp the hanger",
    "Lift the hanger off the rod",
    "Pass hanger from right to left arm",
    "Hook one side of the shirt onto the hanger",
    "Hook the other side of the shirt onto the hanger",
    "Place the hanger on the rod",
)


@dataclasses.dataclass
class ShirtHangDetectorState(DetectorState):
    """State for the shirt-hang online boundary detector."""

    right_gripper_closed_high: bool = False
    right_z_dropped_after_close: bool = False
    right_low_speed_run: int = 0
    left_close_streak: int = 0
    right_open_count_after_left_grab: int = 0
    last_left_is_open: bool = False
    previous_right_is_open: bool = True


class ShirtHangSubtaskDetector(SubtaskDetector):
    """Online detector matching the shirt-hang heuristic boundary ordering."""

    grip_threshold: float = 400.0
    right_grasp_z_threshold: float = 0.30
    right_drop_z_threshold: float = 0.20
    right_stop_speed_threshold: float = 0.015
    right_stop_required_steps: int = 10
    left_close_required_steps: int = 40

    def __init__(self) -> None:
        super().__init__(SHIRT_HANG_SUBTASKS, sample_every_n_steps = 2)

    def initial_state(self) -> ShirtHangDetectorState:
        return ShirtHangDetectorState()

    def observe_sample(self, env_obs: dict, step_idx: int) -> None:
        del step_idx
        state = env_obs["state"]
        right_gripper = float(_last_array(state["right/gripper_pos"])[0])
        left_gripper = float(_last_array(state["left/gripper_pos"])[0])
        right_tcp_pose = _last_array(state["right/tcp_pose"]).astype(np.float32)
        right_tcp_vel = _last_array(state["right/tcp_vel"]).astype(np.float32)
        left_tcp_vel = _last_array(state["left/tcp_vel"]).astype(np.float32)

        right_is_closed = right_gripper <= self.grip_threshold
        left_is_closed = left_gripper <= self.grip_threshold
        left_is_open = not left_is_closed
        right_is_open = not right_is_closed

        state_obj = self._state
        assert isinstance(state_obj, ShirtHangDetectorState)

        if state_obj.current_subtask_index == 0:
            if right_is_closed and right_tcp_pose[2] > self.right_grasp_z_threshold:
                state_obj.right_gripper_closed_high = True
            if state_obj.right_gripper_closed_high and not right_is_open and right_tcp_pose[2] < self.right_drop_z_threshold:
                state_obj.right_z_dropped_after_close = True

        if state_obj.current_subtask_index <= 1:
            right_speed = float(np.linalg.norm(right_tcp_vel[:3]))
            if right_speed < self.right_stop_speed_threshold:
                state_obj.right_low_speed_run += 1
            else:
                state_obj.right_low_speed_run = 0

        if state_obj.current_subtask_index <= 2:
            if left_is_closed:
                state_obj.left_close_streak += 1
            else:
                state_obj.left_close_streak = 0

        if state_obj.current_subtask_index >= 2 and right_is_open:
            if not state_obj.previous_right_is_open:
                state_obj.right_open_count_after_left_grab += 1
        state_obj.previous_right_is_open = right_is_open

        state_obj.last_left_is_open = left_is_open
        state_obj.sampled_steps += 1

    def should_advance_to(self, subtask_index: int) -> bool:
        state = self._state
        assert isinstance(state, ShirtHangDetectorState)
        if subtask_index == 1:
            return state.right_gripper_closed_high and state.right_z_dropped_after_close
        if subtask_index == 2:
            return state.right_low_speed_run >= self.right_stop_required_steps
        if subtask_index == 3:
            return state.left_close_streak >= self.left_close_required_steps
        if subtask_index == 4:
            return state.right_open_count_after_left_grab >= 2
        if subtask_index == 5:
            return state.last_left_is_open
        raise ValueError(f"Unsupported subtask transition target: {subtask_index}")
