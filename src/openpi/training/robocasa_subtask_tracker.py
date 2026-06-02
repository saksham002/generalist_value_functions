"""Subtask boundary tracking for composite RoboCasa tasks.

Atomic RoboCasa tasks are single-segment: ``steps_to_subtask_end`` counts down to the
end of the whole episode. Composite tasks decompose into K ordered subtasks; for those
the K-1 subtask-switch boundaries plus the per-trajectory subtask order are recovered
from the raw trajectory state with a per-dataset heuristic, then converted into a
per-step subtask index. The subtask order is per-trajectory because some tasks admit
multiple valid orderings (e.g. ``arrange_tea``, where the kettle and mug can be picked
in either order).

A dataset is "composite" iff its name does NOT contain the substring "atomic". Every
composite dataset must have a registered ``CompositeSubtaskSpec`` in
``COMPOSITE_SUBTASK_REGISTRY``; an unregistered composite dataset fails loudly.

Raw state layout (16D, pre-conversion — see robocasa_rlds_dataset.py):
    [0:3]   base_position
    [3:7]   base_rotation      (quaternion, xyzw)
    [7:10]  eef_position
    [10:14] eef_rotation       (quaternion, xyzw)
    [14:16] gripper_qpos
"""

from collections.abc import Callable
import dataclasses
import re

import numpy as np

# Raw-state slice holding the two gripper finger positions.
GRIPPER_QPOS_SLICE = slice(14, 16)
# Indices into the raw state for the eef position components.
EEF_Y_INDEX = 8
EEF_Z_INDEX = 9


class ExpectedAnnotationFailure(Exception):
    """Per-episode failure that the tracker is allowed to give up on silently.

    Raised by a ``find_subtasks`` heuristic when the input episode is *known*
    to fall outside the supported domain (e.g. ``wash_fruit_colander`` with
    more than one fruit, which the 3/4-subtask decomposition cannot represent).
    ``compute_subtasks`` catches *only* this exception and returns
    ``annotation_success=False``; every other exception propagates so that
    tracker bugs and unexpected heuristic failures crash loudly.
    """


@dataclasses.dataclass(frozen = True)
class CompositeSubtaskSpec:
    """Subtask definition for one composite RoboCasa dataset.

    Attributes:
        subtasks: Reference subtask description strings of length K_max. Length
            sets the UPPER BOUND on the per-episode subtask count and budgets
            the fixed-shape tensor the data pipeline allocates per episode.
            Three regimes share this representation:
            - Fixed decomposition (e.g. arrange_tea): every episode's
              ``subtask_order`` is a permutation of these strings.
            - Variable count (e.g. wash_fruit_colander with optional spout-turn):
              shorter decompositions return fewer strings; ``compute_subtasks``
              pads ``subtask_order`` to K_max with empty strings.
            - Templated (e.g. weigh_ingredients): per-episode strings substitute
              the ingredient name, so they won't literally match these. These
              entries are illustrative.
        find_subtasks: Maps ``(raw_state [T, 16], language_instruction)`` to
            ``(boundaries, subtask_order)``. Heuristics that don't need the
            instruction simply discard the second argument.
            - ``boundaries`` is an int array of length ``len(subtask_order) - 1``;
              each boundary is the first frame of the next subtask.
            - ``subtask_order`` is a tuple of subtask strings in execution order
              for THIS trajectory; length must be in [1, K_max].
    """

    subtasks: tuple[str, ...]
    find_subtasks: Callable[[np.ndarray, str], tuple[np.ndarray, tuple[str, ...]]]

    @property
    def num_subtasks(self) -> int:
        return len(self.subtasks)


def is_composite_dataset(dataset_name: str) -> bool:
    """A dataset is composite unless its name contains the substring 'atomic'."""
    return "atomic" not in dataset_name


def _find_first_gripper_close(
    gripper_sum: np.ndarray, threshold: float, *, start_index: int = 0,
) -> int:
    """Return the first ``> threshold -> <= threshold`` transition after ``start_index``.

    The gripper signal is large (``> threshold``) when the gripper is open / empty
    and small (``<= threshold``) when it is closed / grasping, so this transition
    marks the first gripper-close (grasp) event strictly after ``start_index``.
    Raises ExpectedAnnotationFailure if no such transition exists.
    """
    for i in range(start_index + 1, gripper_sum.shape[0]):
        if gripper_sum[i - 1] > threshold and gripper_sum[i] <= threshold:
            return i
    raise ExpectedAnnotationFailure(
        f"No gripper close (> {threshold} -> <= {threshold}) found after index {start_index}."
    )


def _find_close_then_release(
    gripper_sum: np.ndarray, start_index: int, threshold: float,
) -> int:
    """Return the first close-then-release event after ``start_index``.

    Scans for the first gripper close (``> threshold -> <= threshold``) and then the
    first gripper release (``<= threshold -> > threshold``) following it, returning
    the release frame index. Raises ExpectedAnnotationFailure if either transition is not found.
    """
    close_index = _find_first_gripper_close(gripper_sum, threshold, start_index = start_index)
    for i in range(close_index + 1, gripper_sum.shape[0]):
        if gripper_sum[i - 1] <= threshold and gripper_sum[i] > threshold:
            return i
    raise ExpectedAnnotationFailure(
        f"No gripper release (<= {threshold} -> > {threshold}) found after close at index {close_index}."
    )


# Gripper-sum threshold separating an open/empty gripper from a closed/grasping one.
_ARRANGE_TEA_GRIPPER_THRESHOLD = 0.05
# Use "doors" (plural) when the EEF y excursion across the whole episode exceeds
# this threshold; "door" (singular) otherwise. Same heuristic as weigh_ingredients.
_ARRANGE_TEA_DOORS_Y_RANGE = 0.65

# Pick-place strings follow the pretrain__atomic__pick_place_{source}_to_{dest}
# format ("Pick the X from the {source} and place it on the {dest}."). The close
# string is templated per episode (door / doors selected by EEF y-range
# heuristic) and built inside ``_arrange_tea_find_subtasks``.
_ARRANGE_TEA_PICK_KETTLE = "Pick the kettle from the counter and place it on the tray."
_ARRANGE_TEA_PICK_MUG = "Pick the mug from the cabinet and place it on the tray."
_ARRANGE_TEA_CANONICAL_SUBTASKS: tuple[str, ...] = (
    _ARRANGE_TEA_PICK_KETTLE,
    _ARRANGE_TEA_PICK_MUG,
    "Close the cabinet doors.",
)


def _arrange_tea_find_subtasks(
    raw_state: np.ndarray, language_instruction: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Boundaries + per-trajectory subtask order for ``target__composite__arrange_tea``.

    Boundaries: the arm releases an object then re-grasps the next one at each
    subtask switch, so a close-then-release event in the gripper signal (sum of
    |finger qpos|) marks a boundary.

    Subtask order: the kettle and mug can be picked in either order. The robot
    reaches UP to grab the mug from the cabinet but stays at or below the starting
    eef-z when reaching for the kettle on the counter, so we compare eef-z at the
    first gripper close to eef-z at trajectory start: if it has risen, the first
    picked object is the mug → swap the first two subtasks in the returned order.
    The third subtask ("Close the cabinet doors") always stays last.
    """
    del language_instruction  # not used; boundaries come from the gripper signal.
    gripper_sum = np.sum(np.abs(raw_state[:, GRIPPER_QPOS_SLICE]), axis = -1)
    first = _find_close_then_release(gripper_sum, 0, _ARRANGE_TEA_GRIPPER_THRESHOLD)
    second = _find_close_then_release(gripper_sum, first, _ARRANGE_TEA_GRIPPER_THRESHOLD)
    boundaries = np.array([first, second], dtype = np.int64)

    first_close_index = _find_first_gripper_close(gripper_sum, _ARRANGE_TEA_GRIPPER_THRESHOLD)
    initial_eef_z = float(raw_state[0, EEF_Z_INDEX])
    eef_z_at_first_close = float(raw_state[first_close_index, EEF_Z_INDEX])

    eef_y = raw_state[:, EEF_Y_INDEX]
    door_word = (
        "doors" if float(eef_y.max() - eef_y.min()) > _ARRANGE_TEA_DOORS_Y_RANGE else "door"
    )
    close_cabinet_subtask = f"Close the cabinet {door_word}."

    if eef_z_at_first_close - initial_eef_z > 0.0:
        subtask_order = (
            _ARRANGE_TEA_PICK_MUG,
            _ARRANGE_TEA_PICK_KETTLE,
            close_cabinet_subtask,
        )
    else:
        subtask_order = (
            _ARRANGE_TEA_PICK_KETTLE,
            _ARRANGE_TEA_PICK_MUG,
            close_cabinet_subtask,
        )
    return boundaries, subtask_order


# Per-boundary gripper thresholds: the colander grasp closes the gripper less
# tightly than the fruit grasp, so the two close-then-release events sit at
# different gripper-sum levels and need their own thresholds.
_WASH_FRUIT_COLANDER_FIRST_THRESHOLD = 0.065
_WASH_FRUIT_COLANDER_SECOND_THRESHOLD = 0.07

# Post-fruit-placement spout-turn detection (using EEF y over [second, ep_len)):
#   Case 2 (LEFT)  — EEF y dips at some j with prefix_max[j] - y[j] > 0.12 AND
#                     suffix_max[j] - y[j] > 0.12, restricted to j >= second + 20.
#   Case 3 (RIGHT) — no valley but EEF y rises by > 0.10 from some j to the suffix max.
#   Case 1 (NONE)  — neither holds.
# When a spout-turn is detected, the third boundary (spout → faucet) is placed at
# the deepest point of the post-fruit y dip (argmin of y_post).
_WASH_FRUIT_COLANDER_SPOUT_VALLEY_DROP = 0.12
_WASH_FRUIT_COLANDER_SPOUT_SUFFIX_DROP = 0.10
_WASH_FRUIT_COLANDER_SPOUT_VALLEY_MIN_OFFSET = 20

# Representative subtasks (case 2/3 form, 4 entries). Case 1 returns the first two
# plus the faucet line and pads to length 4 with an empty string downstream.
_WASH_FRUIT_COLANDER_CANONICAL_SUBTASKS: tuple[str, ...] = (
    "Pick the colander from the counter and place it in the sink.",
    "Pick the fruit from the counter and place it in the colander.",
    "Turn the spout to the left.",
    "Turn on the sink faucet.",
)
_WASH_FRUIT_COLANDER_PICK_COLANDER = _WASH_FRUIT_COLANDER_CANONICAL_SUBTASKS[0]
_WASH_FRUIT_COLANDER_PICK_FRUIT = _WASH_FRUIT_COLANDER_CANONICAL_SUBTASKS[1]
_WASH_FRUIT_COLANDER_TURN_FAUCET = "Turn on the sink faucet."

# Language-instruction template (per a survey of the 507 training episodes):
#   "Put the colander in the sink, put the <FRUITS> in the colander, and
#    turn on the sink faucet and pour water over the colander."
# where <FRUITS> is 1-3 comma-separated fruit names.
_WASH_FRUIT_COLANDER_INSTRUCTION_PATTERN = re.compile(
    r"put the (.+?) in the colander, and turn on the sink faucet"
)


def _wash_fruit_colander_parse_fruits(language_instruction: str) -> list[str]:
    match = _WASH_FRUIT_COLANDER_INSTRUCTION_PATTERN.search(language_instruction)
    if match is None:
        raise ValueError(
            f"Could not parse fruit list from wash_fruit_colander instruction: "
            f"{language_instruction!r}."
        )
    return [f.strip() for f in match.group(1).split(",")]


def _wash_fruit_colander_classify_spout(
    eef_y: np.ndarray, second_boundary: int,
) -> tuple[int, int | None]:
    """Classify the post-fruit-placement region into spout case {1, 2, 3}.

    Returns ``(case, spout_end_index)``. ``spout_end_index`` is None for case 1
    (no spout-turn subtask) and the absolute frame index of the deepest point
    of the y dip for cases 2/3 (used as the spout → faucet boundary).
    """
    y_post = eef_y[second_boundary:]
    if y_post.size <= 1:
        return 1, None

    prefix_max = np.maximum.accumulate(y_post)
    suffix_max = np.maximum.accumulate(y_post[::-1])[::-1]
    diff_prefix = prefix_max - y_post
    diff_suffix = suffix_max - y_post

    if y_post.size > _WASH_FRUIT_COLANDER_SPOUT_VALLEY_MIN_OFFSET:
        case_2_mask = (diff_prefix > _WASH_FRUIT_COLANDER_SPOUT_VALLEY_DROP) & (
            diff_suffix > _WASH_FRUIT_COLANDER_SPOUT_VALLEY_DROP
        )
        case_2_mask[:_WASH_FRUIT_COLANDER_SPOUT_VALLEY_MIN_OFFSET] = False
        if bool(case_2_mask.any()):
            spout_end_local = int(np.argmin(y_post))
            return 2, second_boundary + spout_end_local

    if bool((diff_suffix > _WASH_FRUIT_COLANDER_SPOUT_SUFFIX_DROP).any()):
        spout_end_local = int(np.argmin(y_post))
        return 3, second_boundary + spout_end_local

    return 1, None


def _wash_fruit_colander_find_subtasks(
    raw_state: np.ndarray, language_instruction: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Boundaries + subtask order for ``target__composite__wash_fruit_colander``.

    The dataset's language instructions list 1-3 fruits (comma-separated). The
    canonical decomposition assumes a single fruit, so multi-fruit episodes raise
    here — ``compute_subtasks`` catches the ValueError and reports
    ``annotation_success=False`` so those frames are filtered out.

    Single-fruit episodes always begin: place the colander in the sink → place
    the fruit in the colander. Those two pick-place segments each end on a
    gripper release, found via a close-then-release event in the gripper signal.

    After the fruit is placed, three cases are distinguished from the EEF y
    trajectory in ``[second, ep_len)`` (see ``_wash_fruit_colander_classify_spout``):
    - Case 1: directly "Turn on the sink faucet." (3 subtasks total).
    - Case 2: "Turn the spout to the left." → "Turn on the sink faucet." (4 total).
    - Case 3: "Turn the spout to the right." → "Turn on the sink faucet." (4 total).
    """
    fruits = _wash_fruit_colander_parse_fruits(language_instruction)
    if len(fruits) != 1:
        # Multi-fruit episodes are unsupported by design; raise the
        # caught-by-compute_subtasks "expected" exception so the episode is
        # skipped rather than crashing the data pipeline.
        raise ExpectedAnnotationFailure(
            f"target__composite__wash_fruit_colander multi-fruit episode "
            f"(got {len(fruits)} fruits {fruits}) in {language_instruction!r}."
        )
    gripper_sum = np.sum(np.abs(raw_state[:, GRIPPER_QPOS_SLICE]), axis = -1)
    first = _find_close_then_release(gripper_sum, 0, _WASH_FRUIT_COLANDER_FIRST_THRESHOLD)
    second = _find_close_then_release(gripper_sum, first, _WASH_FRUIT_COLANDER_SECOND_THRESHOLD)

    spout_case, spout_end = _wash_fruit_colander_classify_spout(raw_state[:, EEF_Y_INDEX], second)

    # Per-episode subtask 2 substitutes the actual fruit name (kiwi, apple, ...)
    # parsed from the instruction so the prompt names the specific object.
    pick_fruit_subtask = (
        f"Pick the {fruits[0]} from the counter and place it in the colander."
    )
    if spout_case == 1:
        boundaries = np.array([first, second], dtype = np.int64)
        subtask_order = (
            _WASH_FRUIT_COLANDER_PICK_COLANDER,
            pick_fruit_subtask,
            _WASH_FRUIT_COLANDER_TURN_FAUCET,
        )
    else:
        spout_string = (
            "Turn the spout to the left." if spout_case == 2 else "Turn the spout to the right."
        )
        boundaries = np.array([first, second, spout_end], dtype = np.int64)
        subtask_order = (
            _WASH_FRUIT_COLANDER_PICK_COLANDER,
            pick_fruit_subtask,
            spout_string,
            _WASH_FRUIT_COLANDER_TURN_FAUCET,
        )
    return boundaries, subtask_order


# Language-instruction template for weigh_ingredients (per 504-episode survey):
#   "Pick the <INGREDIENT> and place it on the digital scale for weighing,
#    and close the cabinet."
# INGREDIENT is one of 7 distinct values (canned food, cereal, honey bottle, ...).
_WEIGH_INGREDIENTS_INSTRUCTION_PATTERN = re.compile(
    r"^Pick the (.+?) and place it on the digital scale for weighing, "
    r"and close the cabinet\.?\s*$"
)
# Gripper threshold for the ingredient pick-place release.
_WEIGH_INGREDIENTS_GRIPPER_THRESHOLD = 0.07
# Use "doors" (plural) when the EEF y excursion across the whole episode exceeds
# this threshold; "door" (singular) otherwise. Two-door cabinets force the arm
# to sweep further sideways while closing, giving a larger y range.
_WEIGH_INGREDIENTS_DOORS_Y_RANGE = 0.65

# Representative subtask strings. Ingredient name and door/doors are substituted
# per episode by ``_weigh_ingredients_find_subtasks``; these placeholders set
# K_max = 2 for shape budgeting.
_WEIGH_INGREDIENTS_CANONICAL_SUBTASKS: tuple[str, ...] = (
    "Pick the ingredient from the cabinet and place it on the digital weighing scale.",
    "Close the cabinet doors.",
)


def _weigh_ingredients_parse_ingredient(language_instruction: str) -> str:
    match = _WEIGH_INGREDIENTS_INSTRUCTION_PATTERN.match(language_instruction.strip())
    if match is None:
        raise ValueError(
            f"Could not parse ingredient from weigh_ingredients instruction: "
            f"{language_instruction!r}."
        )
    return match.group(1).strip()


def _weigh_ingredients_find_subtasks(
    raw_state: np.ndarray, language_instruction: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Boundaries + subtask order for ``target__composite__weigh_ingredients``.

    Every episode picks one ingredient from a cabinet and places it on the
    digital weighing scale, then closes the cabinet. The single boundary
    between these two subtasks is the first gripper close-then-release event
    (the moment the ingredient is released on the scale).

    The closing subtask refers to the cabinet's door(s); whether the scene
    has one door or two is not encoded in the language instruction or per-step
    state, but the two-door close requires a larger sideways sweep — so we
    classify from the EEF y excursion across the full episode (see
    ``_WEIGH_INGREDIENTS_DOORS_Y_RANGE``).
    """
    ingredient = _weigh_ingredients_parse_ingredient(language_instruction)
    gripper_sum = np.sum(np.abs(raw_state[:, GRIPPER_QPOS_SLICE]), axis = -1)
    boundary = _find_close_then_release(gripper_sum, 0, _WEIGH_INGREDIENTS_GRIPPER_THRESHOLD)

    eef_y = raw_state[:, EEF_Y_INDEX]
    door_word = (
        "doors" if float(eef_y.max() - eef_y.min()) > _WEIGH_INGREDIENTS_DOORS_Y_RANGE else "door"
    )
    subtask_order = (
        f"Pick the {ingredient} from the cabinet and place it on the digital weighing scale.",
        f"Close the cabinet {door_word}.",
    )
    boundaries = np.array([boundary], dtype = np.int64)
    return boundaries, subtask_order


# Language-instruction template for prepare_coffee (per a 514-episode survey,
# 1 distinct instruction):
#   "Pick the <OBJECT> from the cabinet, place it under the coffee machine
#    dispenser, and press the start button."
# OBJECT is always "mug" in the available data; parsed to keep the tracker
# robust to future instruction variants.
_PREPARE_COFFEE_INSTRUCTION_PATTERN = re.compile(
    r"^Pick the (.+?) from the cabinet, place it under the coffee machine dispenser, "
    r"and press the start button\.?\s*$"
)
# Gripper threshold for the mug pick-place release.
_PREPARE_COFFEE_GRIPPER_THRESHOLD = 0.065
# Subtask 2 canonical text taken verbatim from pretrain__atomic__start_coffee_machine.
_PREPARE_COFFEE_PRESS_BUTTON_SUBTASK = "Press the button on the coffee machine to serve coffee."

# Representative subtasks (K_max = 2). Subtask 1 substitutes the object name per
# episode. Destination phrasing matches pretrain__atomic__coffee_setup_mug
# ("under the coffee machine dispenser"); source stays as "from the cabinet"
# because prepare_coffee episodes actually start in the cabinet.
_PREPARE_COFFEE_CANONICAL_SUBTASKS: tuple[str, ...] = (
    "Pick the mug from the cabinet and place it under the coffee machine dispenser.",
    _PREPARE_COFFEE_PRESS_BUTTON_SUBTASK,
)


def _prepare_coffee_parse_object(language_instruction: str) -> str:
    match = _PREPARE_COFFEE_INSTRUCTION_PATTERN.match(language_instruction.strip())
    if match is None:
        raise ValueError(
            f"Could not parse object from prepare_coffee instruction: {language_instruction!r}."
        )
    return match.group(1).strip()


def _prepare_coffee_find_subtasks(
    raw_state: np.ndarray, language_instruction: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Boundaries + subtask order for ``target__composite__prepare_coffee``.

    Every episode picks an object (always "mug" in the surveyed data) from the
    cabinet, places it under the coffee machine dispenser, and presses the
    start button. The single boundary between the two subtasks is the first
    gripper close-then-release event (same family as wash_fruit_colander and
    weigh_ingredients). Subtask 2's canonical wording is copied from
    ``pretrain__atomic__start_coffee_machine``.
    """
    obj = _prepare_coffee_parse_object(language_instruction)
    gripper_sum = np.sum(np.abs(raw_state[:, GRIPPER_QPOS_SLICE]), axis = -1)
    boundary = _find_close_then_release(gripper_sum, 0, _PREPARE_COFFEE_GRIPPER_THRESHOLD)
    subtask_order = (
        f"Pick the {obj} from the cabinet and place it under the coffee machine dispenser.",
        _PREPARE_COFFEE_PRESS_BUTTON_SUBTASK,
    )
    boundaries = np.array([boundary], dtype = np.int64)
    return boundaries, subtask_order


COMPOSITE_SUBTASK_REGISTRY: dict[str, CompositeSubtaskSpec] = {
    "target__composite__arrange_tea": CompositeSubtaskSpec(
        subtasks = _ARRANGE_TEA_CANONICAL_SUBTASKS,
        find_subtasks = _arrange_tea_find_subtasks,
    ),
    "target__composite__wash_fruit_colander": CompositeSubtaskSpec(
        subtasks = _WASH_FRUIT_COLANDER_CANONICAL_SUBTASKS,
        find_subtasks = _wash_fruit_colander_find_subtasks,
    ),
    "target__composite__weigh_ingredients": CompositeSubtaskSpec(
        subtasks = _WEIGH_INGREDIENTS_CANONICAL_SUBTASKS,
        find_subtasks = _weigh_ingredients_find_subtasks,
    ),
    "target__composite__prepare_coffee": CompositeSubtaskSpec(
        subtasks = _PREPARE_COFFEE_CANONICAL_SUBTASKS,
        find_subtasks = _prepare_coffee_find_subtasks,
    ),
}


def get_subtask_spec(dataset_name: str) -> CompositeSubtaskSpec:
    """Return the CompositeSubtaskSpec for a composite dataset, or raise ValueError."""
    if dataset_name not in COMPOSITE_SUBTASK_REGISTRY:
        raise ValueError(
            f"Composite dataset '{dataset_name}' has no registered CompositeSubtaskSpec. "
            f"Registered composite datasets: {sorted(COMPOSITE_SUBTASK_REGISTRY)}. "
            f"Add a heuristic to COMPOSITE_SUBTASK_REGISTRY in robocasa_subtask_tracker.py."
        )
    return COMPOSITE_SUBTASK_REGISTRY[dataset_name]


def compute_subtasks(
    raw_state: np.ndarray, language_instruction: str, spec: CompositeSubtaskSpec,
) -> tuple[np.ndarray, tuple[str, ...], bool]:
    """Map a trajectory's raw state to a per-step subtask index plus per-trajectory order.

    Runs ``spec.find_subtasks``, validates the result, and converts the K-1 boundaries
    into a per-step int32 subtask index in [0, K). ``subtask_index[i] == j`` means
    frame i belongs to the j-th subtask in ``subtask_order``; ``boundaries[j]`` is
    the first frame of subtask j+1 in ``subtask_order``.

    Args:
        raw_state: Raw trajectory state, shape [T, 16].
        language_instruction: Per-trajectory language instruction. Heuristics that
            don't need it (e.g. arrange_tea) discard it.
        spec: The composite subtask spec for this dataset.

    Returns:
        Tuple of (subtask_index, subtask_order, annotation_success):
            - subtask_index: int32 array of shape [T] with values in [0, K).
            - subtask_order: tuple[str, ...] of length K with the per-trajectory order.
            - annotation_success: False iff the heuristic decomposition failed
              (either ``find_subtasks`` raised, or returned invalid boundaries /
              subtask order). On failure, ``subtask_index`` is filled with zeros
              and ``subtask_order`` is the canonical ``spec.subtasks`` — these
              are placeholders meant to be filtered out downstream.
    """
    num_steps = raw_state.shape[0]
    max_num_subtasks = spec.num_subtasks

    try:
        boundaries, subtask_order = spec.find_subtasks(raw_state, language_instruction)
        boundaries = np.asarray(boundaries)

        if not 1 <= len(subtask_order) <= max_num_subtasks:
            raise ValueError(
                f"subtask_order length must be in [1, {max_num_subtasks}], "
                f"got {len(subtask_order)}."
            )
        num_boundaries = len(subtask_order) - 1
        if boundaries.shape != (num_boundaries,):
            raise ValueError(
                f"boundaries must have length {num_boundaries} (len(subtask_order) - 1), "
                f"got shape {boundaries.shape}."
            )
        if num_boundaries > 0:
            if not np.all(np.diff(boundaries) > 0):
                raise ValueError(f"Subtask boundaries must be strictly increasing, got {boundaries}.")
            if boundaries[0] <= 0 or boundaries[-1] >= num_steps:
                raise ValueError(
                    f"Subtask boundaries must lie strictly inside (0, {num_steps}), got {boundaries}."
                )
    except ExpectedAnnotationFailure:
        # Only "the heuristic deliberately gave up" gets silently converted to
        # annotation_success=False. Everything else (ValueError from invalid
        # boundaries, IndexError, unexpected gripper-pattern failures, etc.)
        # propagates and crashes — those indicate tracker bugs or
        # out-of-distribution episodes worth surfacing.
        return np.zeros(num_steps, dtype = np.int32), tuple(spec.subtasks), False

    subtask_index = np.searchsorted(boundaries, np.arange(num_steps), side = "right")
    # Pad subtask_order with empty strings to the spec-budgeted upper bound; the
    # data pipeline allocates a fixed [K_max] string tensor per episode.
    padded_order = tuple(subtask_order) + ("",) * (max_num_subtasks - len(subtask_order))
    return subtask_index.astype(np.int32), padded_order, True
