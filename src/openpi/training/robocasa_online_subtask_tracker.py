"""Online (per-step, causal) subtask trackers for RoboCasa composite tasks at
eval time. Sibling to the offline ``robocasa_subtask_tracker`` module: same
boundary heuristics, but stateful and incremental so the eval client can
update its current-subtask prompt every env step.

Each tracker is constructed once per episode with the env's language
instruction, then fed one raw-state vector per env step via ``update``. The
current subtask string is exposed via ``current_subtask``. Unlike the offline
tracker, NO forward-looking signal is available, so heuristics the offline
tracker derives from full-episode data (notably ``weigh_ingredients`` door/doors,
which uses ``max(eef_y) - min(eef_y)`` over the entire episode) are
approximated with a running-statistic substitute.

Raw-state layout matches ``robocasa_subtask_tracker``:
    [7:10]  eef_position    (x, y, z)
    [14:16] gripper_qpos    (two-finger)
"""

from __future__ import annotations

import abc
import re

import numpy as np

GRIPPER_QPOS_SLICE = slice(14, 16)
EEF_Y_INDEX = 8
EEF_Z_INDEX = 9


class OnlineSubtaskTracker(abc.ABC):
    """Per-episode causal subtask tracker.

    Lifecycle:
      tracker = SomeTracker(language_instruction)
      for t in range(horizon):
          tracker.update(state_t, t)
          prompt = tracker.current_subtask
          # use prompt in policy infer call
    """

    @abc.abstractmethod
    def update(self, state: np.ndarray, t: int) -> None: ...

    @property
    @abc.abstractmethod
    def current_subtask(self) -> str: ...


def _gripper_event_step(
    prev_open: bool | None, is_open: bool, closed_seen: bool,
) -> tuple[bool | None, bool, bool]:
    """One step of the close-then-release state machine.

    Returns ``(new_prev_open, new_closed_seen, released)``. ``released`` is True
    iff the gripper just transitioned closed → open AFTER having transitioned
    open → closed at some earlier step (i.e. one full close-then-release event
    completed at THIS step).
    """
    released = False
    if prev_open is None:
        return is_open, closed_seen, released
    if prev_open and not is_open:
        closed_seen = True
    elif (not prev_open) and is_open and closed_seen:
        released = True
        closed_seen = False
    return is_open, closed_seen, released


# ---------------------------------------------------------------------------
# weigh_ingredients
# ---------------------------------------------------------------------------

_WEIGH_INGREDIENTS_PATTERN = re.compile(
    r"^Pick the (.+?) and place it on the digital scale for weighing, "
    r"and close the cabinet\.?\s*$"
)


class WeighIngredientsOnlineTracker(OnlineSubtaskTracker):
    """Online weigh_ingredients tracker.

    Mirrors the offline two-subtask decomposition:
      0: "Pick the {ingredient} from the cabinet and place it on the digital weighing scale."
      1: "Close the cabinet {door|doors}."

    Boundary heuristic: first gripper close-then-release at threshold 0.07
    (same threshold the offline tracker uses).

    Door/doors handling: robocasa's ``self.cab`` is a ``SingleCabinet`` (1 door)
    or ``HingeCabinet`` (2 doors). The eval client looks that up and passes
    ``num_doors`` here so the prompt is correct from t=0. ``num_doors`` is
    required — pass it explicitly at construction.
    """

    _GRIPPER_THRESHOLD = 0.07

    def __init__(
        self, language_instruction: str, num_doors: int | None = None,
    ) -> None:
        if num_doors is None:
            raise ValueError(
                "WeighIngredientsOnlineTracker requires num_doors (1 or 2); got None."
            )
        match = _WEIGH_INGREDIENTS_PATTERN.match(language_instruction.strip())
        if match is None:
            raise ValueError(
                f"WeighIngredientsOnlineTracker could not parse instruction: "
                f"{language_instruction!r}."
            )
        ingredient = match.group(1).strip()
        self._pick_subtask = (
            f"Pick the {ingredient} from the cabinet and place it on the digital weighing scale."
        )
        self._idx = 0
        self._prev_open: bool | None = None
        self._closed_seen = False
        self._num_doors = num_doors

    def update(self, state: np.ndarray, t: int) -> None:
        if self._idx >= 1:
            return
        gripper_sum = float(np.sum(np.abs(state[GRIPPER_QPOS_SLICE])))
        is_open = gripper_sum > self._GRIPPER_THRESHOLD
        self._prev_open, self._closed_seen, released = _gripper_event_step(
            self._prev_open, is_open, self._closed_seen,
        )
        if released:
            self._idx = 1

    @property
    def current_subtask(self) -> str:
        if self._idx == 0:
            return self._pick_subtask
        door_word = "doors" if self._num_doors > 1 else "door"
        return f"Close the cabinet {door_word}."


# ---------------------------------------------------------------------------
# prepare_coffee
# ---------------------------------------------------------------------------

_PREPARE_COFFEE_PATTERN = re.compile(
    r"^Pick the (.+?) from the cabinet, place it under the coffee machine dispenser, "
    r"and press the start button\.?\s*$"
)


class PrepareCoffeeOnlineTracker(OnlineSubtaskTracker):
    """Online prepare_coffee tracker.

    Mirrors the offline two-subtask decomposition:
      0: "Pick the {object} from the cabinet and place it under the coffee machine dispenser."
      1: "Press the button on the coffee machine to serve coffee."

    Boundary heuristic: first gripper close-then-release at threshold 0.065
    (matches the offline tracker).
    """

    _GRIPPER_THRESHOLD = 0.065
    _PRESS_SUBTASK = "Press the button on the coffee machine to serve coffee."

    def __init__(self, language_instruction: str) -> None:
        match = _PREPARE_COFFEE_PATTERN.match(language_instruction.strip())
        if match is None:
            raise ValueError(
                f"PrepareCoffeeOnlineTracker could not parse instruction: "
                f"{language_instruction!r}."
            )
        obj = match.group(1).strip()
        self._pick_subtask = (
            f"Pick the {obj} from the cabinet and place it under the coffee machine dispenser."
        )
        self._idx = 0
        self._prev_open: bool | None = None
        self._closed_seen = False

    def update(self, state: np.ndarray, t: int) -> None:
        if self._idx >= 1:
            return
        gripper_sum = float(np.sum(np.abs(state[GRIPPER_QPOS_SLICE])))
        is_open = gripper_sum > self._GRIPPER_THRESHOLD
        self._prev_open, self._closed_seen, released = _gripper_event_step(
            self._prev_open, is_open, self._closed_seen,
        )
        if released:
            self._idx = 1

    @property
    def current_subtask(self) -> str:
        return self._pick_subtask if self._idx == 0 else self._PRESS_SUBTASK


# ---------------------------------------------------------------------------
# arrange_tea
# ---------------------------------------------------------------------------

_ARRANGE_TEA_PICK_KETTLE = "Pick the kettle from the counter and place it on the tray."
_ARRANGE_TEA_PICK_MUG = "Pick the mug from the cabinet and place it on the tray."
# Env steps at native 20 Hz (env_action_fps == native_fps for RoboCasa), so the
# first second of the episode is the first 20 env steps.
_ARRANGE_TEA_FIRST_SECOND_STEPS = 20
# eef-z rise over the first second separating mug-first (arm reaches UP into the
# cabinet) from kettle-first (arm only descends to the counter). Calibrated on
# the 498-episode arrange_tea dataset: kettle-first maxRise <= 0.027, mug-first
# >= 0.043 for all but one dip-then-reach episode.
_ARRANGE_TEA_Z_RISE_THRESHOLD = 0.035


class ArrangeTeaOnlineTracker(OnlineSubtaskTracker):
    """Online arrange_tea tracker.

    arrange_tea picks the kettle (from the counter) and the mug (from the
    cabinet) onto a tray in EITHER order, then closes the cabinet. The
    whole-episode instruction is a fixed template that always lists kettle-first,
    so the order can't be read from language — it must come from the trajectory.

    Order decision (causal): the arm reaches UP into the cabinet for the mug but
    only descends to the counter for the kettle. Over the first second
    (``_ARRANGE_TEA_FIRST_SECOND_STEPS`` env steps) the peak eef-z is tracked; if
    it rises more than ``_ARRANGE_TEA_Z_RISE_THRESHOLD`` above the t=0 value, the
    first picked object is the mug, else the kettle. ``current_subtask`` is
    ``"null"`` until the decision is made at the end of the first second.

    Subtask advancement: after the order is fixed, each gripper close-then-release
    (threshold 0.05, matching the offline tracker's boundaries) advances to the
    next subtask:
      0: first pick (mug or kettle)
      1: second pick (the other)
      2: "Close the cabinet door(s)."
    """

    _GRIPPER_THRESHOLD = 0.05

    def __init__(self, language_instruction: str, num_doors: int | None = None) -> None:
        del language_instruction  # fixed template; the order comes from the trajectory.
        door_word = "door" if num_doors == 1 else "doors"
        self._close_subtask = f"Close the cabinet {door_word}."
        self._z0: float | None = None
        self._max_z: float | None = None
        self._order: tuple[str, str] | None = None
        self._idx = 0
        self._prev_open: bool | None = None
        self._closed_seen = False

    def update(self, state: np.ndarray, t: int) -> None:
        z = float(state[EEF_Z_INDEX])
        if t == 0:
            self._z0 = z
            self._max_z = z

        # Track the peak eef-z over the first-second window, then commit the order
        # once the window has fully elapsed.
        if self._order is None:
            self._max_z = max(self._max_z, z)
            if t >= _ARRANGE_TEA_FIRST_SECOND_STEPS:
                if self._max_z - self._z0 > _ARRANGE_TEA_Z_RISE_THRESHOLD:
                    self._order = (_ARRANGE_TEA_PICK_MUG, _ARRANGE_TEA_PICK_KETTLE)
                else:
                    self._order = (_ARRANGE_TEA_PICK_KETTLE, _ARRANGE_TEA_PICK_MUG)

        if self._idx < 2:
            gripper_sum = float(np.sum(np.abs(state[GRIPPER_QPOS_SLICE])))
            is_open = gripper_sum > self._GRIPPER_THRESHOLD
            self._prev_open, self._closed_seen, released = _gripper_event_step(
                self._prev_open, is_open, self._closed_seen,
            )
            if released:
                self._idx += 1

    @property
    def current_subtask(self) -> str:
        if self._order is None:
            return "null"
        ordered = (self._order[0], self._order[1], self._close_subtask)
        return ordered[min(self._idx, 2)]


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------


def get_online_tracker(
    env_name: str,
    language_instruction: str,
    *,
    num_doors: int | None = None,
) -> OnlineSubtaskTracker | None:
    """Dispatch to the right tracker by RoboCasa env name (case-insensitive
    substring match). Returns ``None`` for envs we don't have a tracker for
    yet — callers should then fall back to a fixed per-episode prompt.

    ``num_doors`` is consumed by trackers that produce door/doors prompts
    (``WeighIngredientsOnlineTracker`` and ``ArrangeTeaOnlineTracker``); other
    trackers ignore it. Look it up at the call site via
    ``isinstance(env.cab, SingleCabinet)`` vs ``HingeCabinet`` (robocasa) — see
    examples/robocasa/main.py.
    """
    name = env_name.lower()
    if "weigh" in name and "ingredient" in name:
        return WeighIngredientsOnlineTracker(language_instruction, num_doors = num_doors)
    if "prepare" in name and "coffee" in name:
        return PrepareCoffeeOnlineTracker(language_instruction)
    if "arrange" in name and "tea" in name:
        return ArrangeTeaOnlineTracker(language_instruction, num_doors = num_doors)
    return None
