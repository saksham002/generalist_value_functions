"""Predefined RoboCasa RLDS dataset groupings for joint training.

Each entry of ``ROBOCASA_DATASETS`` is a tuple of ``RLDSDataset`` objects ready to
plug into ``RLDSRoboCasaDataConfig.datasets`` for a particular training regime.
Weights within a group are precomputed as ``N_i / Σ N_j`` from per-dataset step
counts, so ``sample_from_datasets`` yields per-step-uniform draws over the union
(no under/over-sampling of small datasets).
"""

import openpi.training.rlds_dataset as rlds_dataset


def _step_weighted(step_counts: dict[str, int]) -> tuple[rlds_dataset.RLDSDataset, ...]:
    """Build a tuple of RLDSDataset entries with weights ∝ step counts."""
    total = sum(step_counts.values())
    return tuple(
        rlds_dataset.RLDSDataset(name = name, version = "1.0.0", weight = count / total)
        for name, count in step_counts.items()
    )


# Per-dataset step counts for the RoboCasa zero-base-motion atomic pretraining
# split (49 datasets). Source: pretrain_zero_base_motion_steps.csv.
_PRETRAIN_ZERO_BASE_MOTION_ATOMIC_STEPS: dict[str, int] = {
    "pretrain__atomic__adjust_toaster_oven_temperature": 21328,
    "pretrain__atomic__adjust_water_temperature": 20953,
    "pretrain__atomic__cheesy_bread": 31141,
    "pretrain__atomic__close_cabinet": 27754,
    "pretrain__atomic__close_dishwasher": 15781,
    "pretrain__atomic__close_drawer": 15670,
    "pretrain__atomic__close_electric_kettle_lid": 7530,
    "pretrain__atomic__close_fridge": 26888,
    "pretrain__atomic__close_fridge_drawer": 14946,
    "pretrain__atomic__close_oven": 18230,
    "pretrain__atomic__close_toaster_oven_door": 19815,
    "pretrain__atomic__coffee_serve_mug": 16921,
    "pretrain__atomic__coffee_setup_mug": 23636,
    "pretrain__atomic__make_iced_coffee": 29048,
    "pretrain__atomic__open_blender_lid": 20124,
    "pretrain__atomic__open_cabinet": 37492,
    "pretrain__atomic__open_dishwasher": 18086,
    "pretrain__atomic__open_drawer": 20488,
    "pretrain__atomic__open_electric_kettle_lid": 10928,
    "pretrain__atomic__open_fridge": 33138,
    "pretrain__atomic__open_oven": 15555,
    "pretrain__atomic__open_toaster_oven_door": 15469,
    "pretrain__atomic__pick_place_cabinet_to_counter": 20201,
    "pretrain__atomic__pick_place_counter_to_blender": 38892,
    "pretrain__atomic__pick_place_counter_to_cabinet": 24225,
    "pretrain__atomic__pick_place_counter_to_sink": 22410,
    "pretrain__atomic__pick_place_counter_to_stove": 24039,
    "pretrain__atomic__pick_place_counter_to_toaster_oven": 24313,
    "pretrain__atomic__pick_place_fridge_drawer_to_shelf": 26396,
    "pretrain__atomic__pick_place_fridge_shelf_to_drawer": 27047,
    "pretrain__atomic__pick_place_sink_to_counter": 26397,
    "pretrain__atomic__pick_place_stove_to_counter": 23003,
    "pretrain__atomic__pick_place_toaster_oven_to_counter": 19323,
    "pretrain__atomic__pick_place_toaster_to_counter": 26907,
    "pretrain__atomic__preheat_oven": 21102,
    "pretrain__atomic__slide_dishwasher_rack": 19052,
    "pretrain__atomic__slide_oven_rack": 23958,
    "pretrain__atomic__slide_toaster_oven_rack": 11496,
    "pretrain__atomic__start_coffee_machine": 13722,
    "pretrain__atomic__turn_off_microwave": 15233,
    "pretrain__atomic__turn_off_sink_faucet": 12309,
    "pretrain__atomic__turn_off_stove": 32741,
    "pretrain__atomic__turn_on_blender": 11698,
    "pretrain__atomic__turn_on_electric_kettle": 12460,
    "pretrain__atomic__turn_on_microwave": 14010,
    "pretrain__atomic__turn_on_sink_faucet": 23795,
    "pretrain__atomic__turn_on_toaster": 10042,
    "pretrain__atomic__turn_on_toaster_oven": 17051,
    "pretrain__atomic__turn_sink_spout": 11130,
}


PRETRAIN_ZERO_BASE_MOTION_ATOMIC: tuple[rlds_dataset.RLDSDataset, ...] = _step_weighted(
    _PRETRAIN_ZERO_BASE_MOTION_ATOMIC_STEPS,
)


# Per-dataset step counts for the 4 target composite tasks used by the pi-0.5
# joint composite fine-tune. Counts are over the train split, measured locally
# (sum of step cardinalities across all episodes).
_TARGET_COMPOSITE_JOINT_STEPS: dict[str, int] = {
    # Weights are derived by _step_weighted (count / total, total = 1_077_065).
    "target__composite__prepare_coffee": 279534,     # weight 0.259533
    "target__composite__weigh_ingredients": 299575,  # weight 0.278140
    "target__composite__arrange_tea": 497956,        # weight 0.462327
}


TARGET_COMPOSITE_JOINT: tuple[rlds_dataset.RLDSDataset, ...] = _step_weighted(
    _TARGET_COMPOSITE_JOINT_STEPS,
)


# Registry of RoboCasa dataset groupings. Add new groups (e.g. composite splits,
# target__atomic__*) by appending here; downstream configs reference by index or name.
ROBOCASA_DATASETS: list[tuple[rlds_dataset.RLDSDataset, ...]] = [
    PRETRAIN_ZERO_BASE_MOTION_ATOMIC,
    TARGET_COMPOSITE_JOINT,
]
