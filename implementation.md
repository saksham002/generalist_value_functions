# RLDS Dataset Classes — fps / `td_n` / `action_horizon` Conventions

This repository ships three RLDS dataset subclasses of `BaseRldsDataset`
(`src/openpi/training/rlds_dataset.py`):

| File | Class | Used by configs |
| --- | --- | --- |
| `src/openpi/training/robocoin_rlds_dataset.py` | `RoboCoinRldsDataset` | `robocoin_bimanual_*` (sim, joint or EEF) |
| `src/openpi/training/hdf5_rlds_dataset.py` | `Hdf5RldsDataset` | `real_hang_*` (real HDF5, single subtask) |
| `src/openpi/training/robocasa_rlds_dataset.py` | `RoboCasaRldsDataset` | `robocasa_*` (sim, FPS-interpolated) |

All three feed the same downstream pipeline (norm-stat → clip → model
transforms → `RLDSDataLoader`) and emit dict batches when `critic_mode=True` or
`(Observation, Actions)` tuples when `critic_mode=False`. Their job is to take
the raw RLDS trajectory, optionally rescale time, build action chunks, attach
`steps_to_subtask_end`-derived RL fields (`mc_return`, `reward`, `termination`,
`td_discount`, `action_mask`, `next_action_mask`, `next_observation`,
`next_actions`), and apply per-frame masking.

The classes differ mainly in **(a)** what native fps the source data has,
**(b)** what reference rate `td_n` and `action_horizon` are denominated in,
and **(c)** how those values get mapped to native-step counts inside the
dataset. The rest of this document spells those out.

## 1. The shared underlying convention

The discount factor across all three classes is anchored to an "MDP at
**150 Hz**". A canonical 1-second window therefore covers 150 abstract MDP
time units, and every class computes:

```
exp_per_step(fps) = 150 / fps         # 5.0 at 30 Hz, 3.0 at 50 Hz, 2.5 at 60 Hz
mc_return         = discount ** (exp_per_step * steps_to_subtask_end)
td_reward         = discount ** (exp_per_step * steps_to_subtask_end)   # when termination
td_discount       = discount ** (exp_per_step * td_n_native)
                  = discount ** (150 * seconds_per_chunk)
```

`td_n_native` is the **native-step** equivalent of the configured `td_n`. The
classes pick `td_n_native` differently because each class denominates `td_n`
relative to a different reference fps (see Section 3 below). Once
`td_n_native` is converted, `exp_per_step * td_n_native` always collapses to
the same MDP-time count for any given physical 1-second window — so the
`td_discount` magnitudes for "1 second" configs are equivalent across families:

```
RoboCOIN  (td_n in 50-Hz units): exp=3.0 * td_n  → discount^150 at td_n=50
HDF5      (td_n in 60-Hz units): exp=2.5 * td_n  → discount^150 at td_n=60
RoboCasa  (td_n in 50-Hz units): exp=3.0 * td_n  → discount^150 at td_n=50
```

The same is true for `action_horizon`: it is always denominated in the same
reference-fps units as `td_n`, and the dataset materialises a chunk of that
many slots regardless of native fps (slots past the fps-native validity
boundary are masked out via `action_mask`).

## 2. Native fps per class

| Class | Native fps of the raw RLDS trajectory | Allowed fps inside the dataset |
| --- | --- | --- |
| `RoboCoinRldsDataset` | 30 (most sim configs) or 50 (chunk-wise-delta configs) | 30, 50 |
| `Hdf5RldsDataset` | 60 (real_hang capture rate) | 60 (raw) or 30 (post `subsample=True`) |
| `RoboCasaRldsDataset` | 20 (typical) or whatever `native_fps` says | 30 (via `_interpolate_trajectory` to `target_fps=30`) |

- **RoboCOIN**: reads the per-trajectory `fps` metadata; supports two cases
  (`fps==30` and the trailing `else` which is `fps==50`). The pre-refactor
  code also had an `fps==60` arm; that path is dead for sim and was removed.
- **HDF5**: the raw data is always 60 Hz. When the config sets
  `subsample=True`, `Hdf5RldsDataset._subsample_trajectory` halves the
  trajectory (drops every other state-side step, takes the action at index
  `[1::2]` so each kept action sits at the midpoint between two surviving
  states) and rewrites `traj_metadata.episode_metadata.fps` from 60 to 30.
  All downstream code then sees `fps=30`. When `subsample=False`, the data
  flows through at `fps=60` and the class asserts that the source is 60 Hz.
- **RoboCasa**: the raw data is at `native_fps` (declared on the config),
  but `RoboCasaRldsDataset._interpolate_trajectory` always resamples to
  `target_fps=30`. Per-dim interpolation rules (linear / quat-slerp / nearest
  for images) come from `state_action_spaces.InterpolationConfig`. The class
  refuses any `target_fps != 30`.

## 3. `td_n` and `action_horizon` reference fps

The reference fps determines how the configured integers (`td_n=50`,
`action_horizon=30`, `td_n=60`, etc.) translate to native step counts.

| Class | Reference fps for `td_n` / `action_horizon` | Constraint | `td_n_native` formula |
| --- | --- | --- | --- |
| `RoboCoinRldsDataset` | 50 Hz | both must be `% 5 == 0` | `tf.where(fps==30, 3*td_n//5, td_n)` |
| `Hdf5RldsDataset` | 60 Hz | `action_chunk_size in {30, 60}`, `td_n % 5 == 0` | `tf.where(fps==30, td_n//2, td_n)` |
| `RoboCasaRldsDataset` | 50 Hz | both must be `% 5 == 0` | `3*td_n//5` (hard-coded; target_fps is always 30) |

Concretely:

- **RoboCOIN**: `td_n=50` is "1 second" because at the canonical 50 Hz that's
  exactly 50 native steps. At `fps=30` (the actual rate of most sim configs)
  the dataset converts to `td_n_native = 3*50/5 = 30` native steps — still
  1 second. At `fps=50` no conversion is applied (`td_n_native = td_n`).
- **HDF5**: `td_n=60` is "1 second" because at the canonical 60 Hz that's
  exactly 60 native steps. At `fps=60` (raw, `subsample=False`) the dataset
  uses `td_n` as is. At `fps=30` (post `subsample=True`) it uses `td_n//2 =
  30` native steps. Both paths still represent 1 second of trajectory.
- **RoboCasa**: `td_n=50` is "1 second" using the same 50-Hz reference as
  RoboCOIN. After the mandatory 30-Hz interpolation, `td_n_native = 3*50/5 =
  30` native (post-interpolation) steps — again 1 second. RoboCasa hardcodes
  the fps=30 path because that's the only target it supports.

The `filter_n` knob (drop frames whose `steps_to_subtask_end` is below this
many native steps) is denominated in the same reference fps and is scaled the
same way.

## 4. `action_chunk_size`

In all three classes the dataset emits a chunk of exactly `action_chunk_size`
action slots (a fixed shape `[batch, action_chunk_size, action_dim]`),
regardless of native fps. Only the first `valid_fps_slots` of those positions
are marked True in `action_mask`:

| Class | `valid_fps_slots` at fps=30 | `valid_fps_slots` at fps=50 / 60 |
| --- | --- | --- |
| `RoboCoinRldsDataset` | `3 * action_chunk_size // 5` | `action_chunk_size` |
| `Hdf5RldsDataset` | `action_chunk_size // 2` | `action_chunk_size` (fps=60) |
| `RoboCasaRldsDataset` | n/a (only emits fps=30 batches) | n/a |

So a config like `action_horizon=50` on RoboCOIN at fps=30 produces a
50-slot chunk where the first 30 are valid (= 1 second), and an
`action_horizon=60` HDF5 config at fps=30 produces a 60-slot chunk where the
first 30 are valid (= 1 second). RoboCasa always emits chunks at 30 Hz where
all `action_chunk_size` slots are valid.

The remaining `action_chunk_size - valid_fps_slots` positions are filled by
clamping `actions[frame_idx + offset]` to `actions[traj_len - 1]` (via
`_chunk_indices` in `BaseRldsDataset`). They are masked out of every loss /
gradient via `action_mask`. When the config sets
`replace_boundary_actions=True`, the downstream `ReplaceMaskedActions` model
transform additionally overwrites those slots with the **last valid** action
so masked tail values don't leak into anything that ignores the mask.

## 5. `action_mask` / `next_action_mask` semantics

Three knobs control how `action_mask` is set inside the dataset:

| Knob | Behaviour |
| --- | --- |
| `variable_horizon=True` | Per-step random k-sampling (RoboCOIN / HDF5 only). `action_mask = offsets < k_native`. Used by `robocoin_bimanual_paligemma_q_sarsa_variable_horizon`. |
| `mask_boundary_actions=True` (or `replace_boundary_actions=True`, which sets it via `or`) | Per-trajectory `action_mask = offsets <= steps_to_subtask_end`. Action positions past the subtask boundary are masked. `next_action_mask = offsets <= steps - td_n_native`. |
| Both `False` (and `variable_horizon=False`) | `action_mask` reset to all-True, then fps-clamped to `valid_fps_slots`. The sse-derived mask is **discarded**. |

After whichever branch above ran, both classes (RoboCOIN and HDF5) apply the
fps-clamp:

```python
fps_mask_30 = tf.sequence_mask(valid_fps_slots, action_chunk_size)
action_mask = tf.where(is_30fps, action_mask & fps_mask_30, action_mask)
```

So the trailing positions past `valid_fps_slots` are **always** False on
fps=30 batches regardless of which branch above produced the upstream mask.
This is why `action_mask` on a `subsample=True` HDF5 config is always
"30 True + 30 False" for `action_chunk_size=60` — independent of
`steps_to_subtask_end`.

RoboCasa skips the mask-boundary path entirely (it only emits fps=30 batches
where `valid_fps_slots == action_chunk_size`, so there's nothing to clamp) and
builds `action_mask` directly from `_build_action_mask(traj_len)`.

## 6. Subtask handling

| Class | Subtasks per episode | Frame-level handling |
| --- | --- | --- |
| `RoboCoinRldsDataset` | up to 5 (`subtask_1..5`, `first_null_index`) | per-frame random subtask sampling, plus `include_subtask` filter that drops `static`/`abnormal`/null subtasks. `subtask_prompt_mode` controls the prompt (`subtask_only`, `all_subtasks`, `task_description_*`). |
| `Hdf5RldsDataset` | 1 (`subtask_1` only) | no sampling, no `include_subtask` filter. `prompt_mode` is a binary choice (`subtask` ↔ `task_description`). |
| `RoboCasaRldsDataset` | 1 (`subtask_1` only) | no sampling. `subtask_1` is used as the prompt verbatim. |

A consequence of HDF5 dropping the `include_subtask` filter is that the
filtered-frame stream is denser than RoboCOIN's would be on the same real_hang
data — static / abnormal frames flow through.

## 7. RL field overrides

`BaseRldsDataset._apply_rl_fields` is the hook that critic-mode datasets use
to attach RL-specific outputs.

- **RoboCOIN** does **not** override `_apply_rl_fields`. It computes the RL
  fields per-frame inside `frame_transforms` (after the multi-subtask
  `sampled_idx` is picked) because the choice of subtask affects which
  `steps_to_subtask_end` value the frame ends up with.
- **HDF5** overrides `_apply_rl_fields`. Because there's a single subtask,
  the RL fields are deterministic per step and can be precomputed at
  trajectory level. `frame_transforms` just gathers those per-step scalars
  into the per-frame output.
- **RoboCasa** also overrides `_apply_rl_fields`. Like HDF5 it has a single
  subtask, so it precomputes `mc_return` / `td_discount` / termination /
  reward / `next_observation` / `next_actions_raw` at trajectory level using
  its hardcoded fps=30 constants.

This is also where the "use the base method as much as possible" rule
manifests: both HDF5 and RoboCasa override `_apply_rl_fields` and leave the
rest of `_prepare_trajectory` / `_chunk_actions` to `BaseRldsDataset`, while
RoboCOIN keeps a much larger custom `_prepare_trajectory` to handle the
multi-subtask sampling.

## 8. Quick reference

| Field | RoboCOIN | HDF5 | RoboCasa |
| --- | --- | --- | --- |
| Native fps | 30 or 50 | 60 (raw) / 30 (subsample) | 20 → 30 (interpolated) |
| Reference fps for `td_n`, `action_horizon` | 50 | 60 | 50 |
| `td_n_native` at fps=30 | `3 * td_n // 5` | `td_n // 2` | `3 * td_n // 5` (only branch) |
| `td_n_native` at fps=50 | `td_n` | n/a | n/a |
| `td_n_native` at fps=60 | n/a | `td_n` | n/a |
| `exp_per_step` (MDP=150 Hz) | 5.0 (fps=30) / 3.0 (fps=50) | 5.0 (fps=30) / 2.5 (fps=60) | 5.0 (only) |
| `td_discount` (non-variable) | `discount ** (3.0 * td_n)` | `discount ** (2.5 * td_n)` | `discount ** (exp_per_step * td_n_native)` |
| Typical `td_n` (= 1 sec) | 50 | 60 | 50 |
| Typical `action_horizon` | 50 (or 25 = 0.5 s) | 60 | 50 |
| Chunk dim = `action_chunk_size` slots | yes | yes | yes |
| Multi-subtask | yes (up to 5) | no | no |
| `replace_boundary_actions` semantics | tail masked / replaced by last valid action | same | same (when configured) |
| `subsample` flag | n/a | drops every other state / midpoint action; halves `fps` to 30 | n/a (interpolation instead) |
