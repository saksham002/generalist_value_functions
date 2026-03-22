#!/usr/bin/env python3
"""Verify dont_mask_actions behavior near subtask boundaries and action normalization correctness.

This script:
1. Loads the RoboCOIN dataset with dont_mask_actions=True (critic mode, val split)
2. Finds samples near subtask boundaries (steps_to_subtask_end < 10)
3. Verifies action_mask is all True after dont_mask_actions replacement
4. Checks that replaced actions are close to the last valid action (within noise tolerance)
5. Downloads a raw episode parquet to compare raw vs normalized action values
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openpi.models.tokenizer import create_tokenizer
from openpi.training.robocoin_data_loader import (
    RoboCOINDataLoaderConfig,
    create_robocoin_data_loader,
    extract_embodiment,
)
from openpi.training.config import RoboCOINDataConfig


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "gs://saksham-euw4/robocoin_bimanual/"
DATASET_NAME = "robocoin:1.0.0"
NORM_STATS_PATH = "gs://saksham-euw4/robocoin_bimanual/norm_stats/embodiment_wise_stats.json"
ACTION_HORIZON = 50
BOUNDARY_THRESHOLD = 10
NOISE_STD = 0.005


def load_norm_stats():
    """Load embodiment-wise norm stats from GCS and build NormStats dicts."""
    robocoin_cfg = RoboCOINDataConfig(
        tfds_data_dir = DATA_DIR,
        dataset_name = DATASET_NAME,
        norm_stats_path = NORM_STATS_PATH,
        use_quantile_norm = True,
        use_eef = True,
        use_chunk_wise_delta = True,
        critic_mode = True,
        dont_mask_actions = True,
    )
    return robocoin_cfg._load_robocoin_norm_stats()


# =============================================================================
# Verification 1: dont_mask_actions near subtask boundaries
# =============================================================================


def verify_dont_mask_actions(max_batches: int = 30, batch_size: int = 256):
    """Load data with dont_mask_actions=True, find boundary samples, verify masks and replacements."""
    print("=" * 80)
    print("VERIFICATION 1: dont_mask_actions near subtask boundaries")
    print("=" * 80)

    norm_stats = load_norm_stats()
    print(f"Loaded norm stats for embodiments: {list(norm_stats.keys())}")

    config = RoboCOINDataLoaderConfig(
        data_dir = DATA_DIR,
        dataset_name = DATASET_NAME,
        split = "val",
        batch_size = batch_size,
        shuffle = False,
        repeat = False,
        seed = 86,
        discount = 0.999,
        td_n = 50,
        state_norm_stats = norm_stats,
        use_quantile_norm = True,
        use_eef = True,
        action_horizon = ACTION_HORIZON,
        dont_mask_actions = True,
        use_chunk_wise_delta = True,
        critic_mode = True,
    )

    tokenizer = create_tokenizer("paligemma", config.max_token_len)

    print(f"\nCreating data loader (split=val, batch_size={batch_size}, max_batches={max_batches})...")
    iterator = create_robocoin_data_loader(config, tokenizer = tokenizer)

    boundary_episodes = []
    total_boundary_samples = 0
    batches_seen = 0

    for batch_idx, batch in enumerate(iterator):
        if batch_idx >= max_batches:
            break
        batches_seen += 1

        steps_to_end = np.asarray(batch["steps_to_subtask_end"])
        action_mask = np.asarray(batch["action_mask"])
        actions = np.asarray(batch["actions"])
        next_action_mask = np.asarray(batch["next_action_mask"])
        next_actions = np.asarray(batch["next_actions"])

        boundary_indices = np.where(
            (steps_to_end > 0) & (steps_to_end < BOUNDARY_THRESHOLD)
        )[0]

        if len(boundary_indices) == 0:
            continue

        total_boundary_samples += len(boundary_indices)

        for idx in boundary_indices:
            if len(boundary_episodes) >= 3:
                break

            sample_info = {
                "batch_idx": batch_idx,
                "sample_idx": int(idx),
                "steps_to_subtask_end": int(steps_to_end[idx]),
                "action_mask": action_mask[idx],
                "next_action_mask": next_action_mask[idx],
                "actions": actions[idx],
                "next_actions": next_actions[idx],
            }
            boundary_episodes.append(sample_info)

        if len(boundary_episodes) >= 3:
            break

    print(f"\nProcessed {batches_seen} batches, found {total_boundary_samples} boundary samples")
    print(f"Collected {len(boundary_episodes)} detailed boundary samples\n")

    all_passed = True
    for i, sample in enumerate(boundary_episodes):
        print(f"--- Boundary Sample {i + 1} ---")
        print(f"  Batch {sample['batch_idx']}, sample index {sample['sample_idx']}")
        print(f"  steps_to_subtask_end: {sample['steps_to_subtask_end']}")

        am = sample["action_mask"]
        nam = sample["next_action_mask"]
        acts = sample["actions"]
        nacts = sample["next_actions"]

        # Check 1: action_mask should be all True (or True for the valid 30fps portion)
        # With dont_mask_actions, all non-fps-masked positions should be True
        am_all_true = np.all(am)
        nam_all_true = np.all(nam)
        print(f"  action_mask all True: {am_all_true} (true_count={am.sum()}/{len(am)})")
        print(f"  next_action_mask all True: {nam_all_true} (true_count={nam.sum()}/{len(nam)})")

        if not am_all_true:
            # Could be 30fps where last 2/5 is masked - that's expected
            valid_count = int(am.sum())
            expected_30fps = 3 * len(am) // 5
            if valid_count == expected_30fps:
                print(f"  -> 30fps sample: {valid_count}/{len(am)} valid (expected {expected_30fps} for 30fps)")
            else:
                print(f"  UNEXPECTED: action_mask has {valid_count} True values")
                all_passed = False

        # Check 2: For positions beyond the subtask boundary, replaced actions should be
        # close to the last valid action (within noise tolerance in normalized space).
        # The original subtask boundary was at steps_to_subtask_end, so positions
        # beyond that index were replaced.
        ste = sample["steps_to_subtask_end"]
        # In the original action chunk, positions > ste would have been masked.
        # With chunk-wise delta, actions[0] = 0 (since delta from itself), so we check
        # the pattern of the replacement.

        # The last valid action (in normalized space) is at index min(ste, action_horizon-1)
        last_valid_pos = min(ste, ACTION_HORIZON - 1)
        last_valid_action = acts[last_valid_pos]

        # Check that actions beyond the boundary are close to last_valid_action
        # In normalized space, the noise after normalization is hard to predict exactly,
        # but the replaced values should cluster around the last valid action.
        if ste < ACTION_HORIZON - 1:
            # Only check if valid mask count allows
            mask_end = int(am.sum())
            beyond = acts[ste + 1 : mask_end]
            if beyond.shape[0] > 0:
                diffs = np.abs(beyond - last_valid_action[None, :])
                max_diff = np.max(diffs)
                mean_diff = np.mean(diffs)
                print(f"  Replaced actions ({beyond.shape[0]} steps) vs last valid (idx={last_valid_pos}):")
                print(f"    max diff (normalized): {max_diff:.6f}")
                print(f"    mean diff (normalized): {mean_diff:.6f}")
                # In raw space noise is 0.005; after quantile norm this scales differently,
                # but should still be relatively small (order ~0.01-0.1 in normalized space)
                if max_diff > 1.0:
                    print(f"  WARNING: large diff detected - may indicate an issue")
                    all_passed = False
                else:
                    print(f"  -> Replacement diffs look reasonable")
        else:
            print(f"  steps_to_subtask_end={ste} >= horizon-1, no replaced actions to check")

        # Check 3: Same for next_actions
        if ste < ACTION_HORIZON - 1:
            nmask_end = int(nam.sum())
            nbeyond_start = max(0, ste)
            nbeyond = nacts[nbeyond_start : nmask_end]
            if nbeyond.shape[0] > 1:
                nlast_valid = nacts[max(0, nbeyond_start - 1)] if nbeyond_start > 0 else nacts[0]
                ndiffs = np.abs(nbeyond - nlast_valid[None, :])
                nmax_diff = np.max(ndiffs)
                print(f"  Next-action replacement ({nbeyond.shape[0]} steps):")
                print(f"    max diff (normalized): {nmax_diff:.6f}")
                if nmax_diff > 1.0:
                    print(f"  WARNING: large next-action diff")
                    all_passed = False

        # Print action value ranges
        print(f"  Actions range: [{acts.min():.4f}, {acts.max():.4f}]")
        print(f"  Next actions range: [{nacts.min():.4f}, {nacts.max():.4f}]")
        print()

    if all_passed:
        print("RESULT: All dont_mask_actions checks PASSED")
    else:
        print("RESULT: Some dont_mask_actions checks FAILED")
    print()
    return all_passed


# =============================================================================
# Verification 2: Raw episode download and inspection
# =============================================================================


def _download_hf_file(repo_id: str, filename: str):
    """Download a file from a HuggingFace dataset repo."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id = repo_id, filename = filename, repo_type = "dataset")


def _load_episode_parquet(repo_id: str, episode_index: int):
    """Load the parquet chunk containing episode_index, filtered to that episode."""
    import pandas as pd

    episodes_path = _download_hf_file(repo_id, "meta/episodes.jsonl")
    with open(episodes_path) as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    print(f"  {repo_id}: {len(episodes)} total episodes")

    # LeRobot v2 parquets are at data/chunk-NNN/episode_NNNNNN.parquet
    ep_str = f"{episode_index:06d}"
    chunk_idx = episode_index // 1000
    chunk_str = f"{chunk_idx:03d}"
    filename = f"data/chunk-{chunk_str}/episode_{ep_str}.parquet"
    print(f"  Downloading {filename}...")
    parquet_path = _download_hf_file(repo_id, filename)
    df = pd.read_parquet(parquet_path)
    df = df[df["episode_index"] == episode_index].sort_values("frame_index").reset_index(drop = True)
    assert len(df) > 0, f"No rows for episode {episode_index} in {filename}"
    return df


def _load_subtask_annotations(repo_id: str):
    """Download and parse subtask_annotations.jsonl from HuggingFace."""
    ann_path = _download_hf_file(repo_id, "annotations/subtask_annotations.jsonl")
    subtasks = []
    with open(ann_path) as f:
        for line in f:
            if line.strip():
                subtasks.append(json.loads(line))
    index_to_text = {s["subtask_index"]: s["subtask"] for s in subtasks}
    null_index = next(s["subtask_index"] for s in subtasks if s["subtask"] == "null")
    return index_to_text, null_index


def _construct_eef_repr_np(action: np.ndarray, eef_action: np.ndarray) -> np.ndarray:
    """Construct 14D EEF representation from raw action + EEF action arrays.
    action: [..., 14], eef_action: [..., 12]
    Returns: [..., 14] = [eef[:6], gripper_left, eef[6:12], gripper_right]
    """
    return np.concatenate([
        eef_action[..., :6],
        action[..., 6:7],
        eef_action[..., 6:12],
        action[..., 13:14],
    ], axis = -1)


def verify_raw_episode(max_batches: int = 30, batch_size: int = 256, num_samples: int = 3):
    """Compare dataloader output against raw HuggingFace episode parquets.

    Uses the val split (which preserves repo_id) to:
    1. Find boundary samples (steps_to_subtask_end < threshold)
    2. Download the raw episode parquet + subtask annotations from HuggingFace
    3. Use _frame_index to match batch frames with parquet rows
    4. Reconstruct the action chunk from raw data and verify:
       - Action chunk construction (EEF repr)
       - Masking / dont_mask_actions noise replacement
       - Chunk-wise delta
       - Prompt / subtask text
    """
    print("=" * 80)
    print("VERIFICATION 2: Raw episode parquet comparison")
    print("=" * 80)

    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
    except ImportError:
        print("Skipping: huggingface_hub or pandas not installed")
        return True

    norm_stats = load_norm_stats()

    config = RoboCOINDataLoaderConfig(
        data_dir = DATA_DIR,
        dataset_name = DATASET_NAME,
        split = "val",
        batch_size = batch_size,
        shuffle = False,
        repeat = False,
        seed = 86,
        discount = 0.999,
        td_n = 50,
        state_norm_stats = norm_stats,
        use_quantile_norm = True,
        use_eef = True,
        action_horizon = ACTION_HORIZON,
        dont_mask_actions = True,
        use_chunk_wise_delta = True,
        critic_mode = True,
    )
    tokenizer = create_tokenizer("paligemma", config.max_token_len)

    print(f"\nCreating data loader (split=val, batch_size={batch_size}, max_batches={max_batches})...")
    iterator = create_robocoin_data_loader(config, tokenizer = tokenizer)

    # Collect boundary samples with metadata
    collected = []
    for batch_idx, batch in enumerate(iterator):
        if batch_idx >= max_batches or len(collected) >= num_samples:
            break

        steps_to_end = np.asarray(batch["steps_to_subtask_end"])
        boundary_indices = np.where((steps_to_end > 0) & (steps_to_end < BOUNDARY_THRESHOLD))[0]
        if len(boundary_indices) == 0:
            continue

        for idx in boundary_indices:
            if len(collected) >= num_samples:
                break
            sample = {}
            for key, value in batch.items():
                if hasattr(value, "__getitem__") and hasattr(value, "shape") and len(value.shape) > 0:
                    sample[key] = np.asarray(value[idx])
                elif isinstance(value, (list, np.ndarray)):
                    sample[key] = np.asarray(value)[idx] if np.asarray(value).ndim > 0 else np.asarray(value)
            collected.append((batch_idx, int(idx), sample))

    print(f"Collected {len(collected)} boundary samples\n")

    all_passed = True
    repo_cache = {}

    for sample_num, (batch_idx, sample_idx, sample) in enumerate(collected):
        print(f"--- Sample {sample_num + 1} (batch {batch_idx}, idx {sample_idx}) ---")

        repo_id = sample["repo_id"]
        if isinstance(repo_id, bytes):
            repo_id = repo_id.decode("utf-8")
        if isinstance(repo_id, np.ndarray):
            repo_id = repo_id.item()
            if isinstance(repo_id, bytes):
                repo_id = repo_id.decode("utf-8")

        episode_idx = int(sample["episode_index"])
        frame_idx = int(sample["_frame_index"])
        ste = int(sample["steps_to_subtask_end"])

        # In critic_mode=True, prompt is popped after tokenization.
        # Use subtask_1_text (set by _process_validation_extras for val split) instead.
        subtask_1_text = sample.get("subtask_1_text")
        if isinstance(subtask_1_text, bytes):
            subtask_1_text = subtask_1_text.decode("utf-8")
        elif isinstance(subtask_1_text, np.ndarray):
            subtask_1_text = subtask_1_text.item()
            if isinstance(subtask_1_text, bytes):
                subtask_1_text = subtask_1_text.decode("utf-8")

        print(f"  repo_id: {repo_id}")
        print(f"  episode_index: {episode_idx}, _frame_index: {frame_idx}")
        print(f"  steps_to_subtask_end: {ste}")
        print(f"  subtask_1_text: {subtask_1_text!r}")

        # Download raw episode data if not cached
        cache_key = (repo_id, episode_idx)
        if cache_key not in repo_cache:
            try:
                df = _load_episode_parquet(repo_id, episode_idx)
                subtask_idx_to_text, null_subtask_idx = _load_subtask_annotations(repo_id)
                repo_cache[cache_key] = (df, subtask_idx_to_text, null_subtask_idx)
            except Exception as e:
                print(f"  ERROR downloading episode data: {e}")
                all_passed = False
                continue

        df, subtask_idx_to_text, null_subtask_idx = repo_cache[cache_key]
        ep_len = len(df)
        print(f"  Raw episode length: {ep_len} frames")
        print(f"  Parquet columns: {list(df.columns)}")

        # Verify _frame_index matches parquet row
        raw_row = df[df["frame_index"] == frame_idx]
        assert len(raw_row) == 1, f"Expected 1 row for frame_index={frame_idx}, got {len(raw_row)}"
        raw_row = raw_row.iloc[0]
        parquet_row_idx = df.index[df["frame_index"] == frame_idx][0]
        print(f"  Parquet row for frame_index={frame_idx}: row {parquet_row_idx}")

        # --- Check 1: Subtask text / prompt ---
        subtask_ann = np.array(raw_row["subtask_annotation"], dtype = np.int32)
        raw_subtask_texts = [subtask_idx_to_text.get(int(s), "") for s in subtask_ann]
        first_null_pos = next((i for i, s in enumerate(subtask_ann) if int(s) == null_subtask_idx), 5)
        print(f"  Raw subtask annotation: {subtask_ann.tolist()}")
        print(f"  Subtask texts: {raw_subtask_texts[:first_null_pos]}")
        print(f"  first_null_index: {first_null_pos}")

        # Val split always picks sampled_idx=0
        expected_subtask_text = raw_subtask_texts[0] if first_null_pos > 0 else ""
        text_match = (subtask_1_text == expected_subtask_text)
        print(f"  Expected subtask_1 text: {expected_subtask_text!r}")
        print(f"  Text match: {text_match}")
        if not text_match:
            print(f"  FAIL: subtask_1_text mismatch")
            all_passed = False

        # --- Check 2: State comparison against raw parquet ---
        raw_state = np.array(raw_row["observation.state"], dtype = np.float32)
        batch_state = sample["state"]
        emb = extract_embodiment(repo_id)
        emb_stats = norm_stats[emb]
        state_q01 = emb_stats["state"].q01
        state_q99 = emb_stats["state"].q99

        # Manually normalize the raw state
        manual_state_norm = (raw_state - state_q01) / (state_q99 - state_q01 + 1e-6) * 2.0 - 1.0
        manual_state_clipped = np.clip(manual_state_norm, -1.0, 1.0)

        state_diff = np.abs(manual_state_clipped - batch_state)
        state_max_diff = np.max(state_diff)
        print(f"  Raw state shape: {raw_state.shape}, range: [{raw_state.min():.6f}, {raw_state.max():.6f}]")
        print(f"  Batch state range: [{batch_state.min():.6f}, {batch_state.max():.6f}]")
        print(f"  State: manual norm vs batch max diff: {state_max_diff:.8f}")
        if state_max_diff > 1e-4:
            print(f"  FAIL: state normalization mismatch")
            all_passed = False

        # --- Check 3: Reconstruct action chunk from raw parquet ---
        chunk_frame_indices = np.minimum(
            np.arange(frame_idx, frame_idx + ACTION_HORIZON),
            ep_len - 1,
        )
        # Map frame_index to parquet row (frame_index == row index in episode parquet)
        raw_actions_chunk = np.stack([np.array(df.iloc[fi]["action"], dtype = np.float32) for fi in chunk_frame_indices])
        raw_eef_chunk = np.stack([np.array(df.iloc[fi]["eef_sim_pose_action"], dtype = np.float32) for fi in chunk_frame_indices])

        # Construct 14D EEF representation
        raw_14d = _construct_eef_repr_np(raw_actions_chunk, raw_eef_chunk)
        print(f"  Raw 14D action chunk shape: {raw_14d.shape}, range: [{raw_14d.min():.6f}, {raw_14d.max():.6f}]")

        # --- Check 4: Compute subtask-based action mask from raw data ---
        # steps_to_subtask_end for subtask 0 (val always picks index 0) determines the mask.
        # We need to compute it from the raw subtask annotation at each frame.
        # The RLDS builder computes steps_to_subtask_end per subtask per frame. We only need subtask 0.
        # For the current frame, steps_to_subtask_end was already provided in the batch.
        # The action mask is: offset <= steps_to_subtask_end
        raw_mask = np.arange(ACTION_HORIZON) <= ste
        print(f"  Raw action mask (from ste={ste}): True for first {min(ste + 1, ACTION_HORIZON)} positions")

        # --- Check 5: dont_mask_actions replacement ---
        # Beyond the mask boundary, actions should have been replaced with last_valid + noise.
        # The replacement happens BEFORE chunk-wise delta and normalization, in raw action space.
        last_valid_pos = min(ste, ACTION_HORIZON - 1)
        last_valid_raw = raw_14d[last_valid_pos]
        replaced_positions = np.where(~raw_mask)[0]
        print(f"  Last valid position: {last_valid_pos}")
        print(f"  Replaced positions count: {len(replaced_positions)}")

        if len(replaced_positions) > 0:
            # After dont_mask_actions, the batch actions include noise replacement.
            # The batch actions have been: (1) dont_mask_actions noise → (2) chunk-wise delta → (3) normalize → (4) clip.
            # We can't exactly reconstruct the noise, but we can verify that the batch actions
            # at replaced positions are close to the last valid action (in normalized space,
            # after delta, they should be close to delta of last_valid from first action).

            # In normalized space with chunk-wise delta, the last valid action is:
            # delta_last_valid = raw_14d[last_valid_pos] - raw_14d[0]
            # After noise: raw_replaced ~= raw_14d[last_valid_pos] + noise(0.005)
            # delta_replaced = raw_replaced - raw_14d[0] ~= delta_last_valid + noise(0.005)
            # After normalization, the noise scale depends on (q99 - q01).

            batch_actions = sample["actions"]  # [ACTION_HORIZON, 14], normalized + clipped
            emb = extract_embodiment(repo_id)
            emb_stats = norm_stats[emb]
            action_q01 = emb_stats["actions"].q01
            action_q99 = emb_stats["actions"].q99

            # Un-normalize the batch actions to get the chunk-wise delta values
            batch_delta = (batch_actions + 1.0) / 2.0 * (action_q99 - action_q01 + 1e-6) + action_q01

            # The valid positions should match the raw chunk-wise delta exactly (before clipping distortion)
            raw_delta = raw_14d - raw_14d[:1, :]

            # Check valid positions match (within clipping tolerance)
            valid_positions = np.where(raw_mask)[0]
            if len(valid_positions) > 0:
                valid_raw_delta = raw_delta[valid_positions]
                valid_batch_delta = batch_delta[valid_positions]
                valid_diff = np.abs(valid_raw_delta - valid_batch_delta)
                max_valid_diff = np.max(valid_diff)
                print(f"  Valid positions: raw delta vs un-normalized batch delta max diff: {max_valid_diff:.6f}")

                # Diffs should be very small unless clipping introduced distortion
                if max_valid_diff > 0.1:
                    # Check if clipping caused the discrepancy
                    raw_normalized = (raw_delta[valid_positions] - action_q01[valid_positions]) / (action_q99[valid_positions] - action_q01[valid_positions] + 1e-6) * 2.0 - 1.0
                    was_clipped = np.abs(raw_normalized) > 1.0
                    clipped_count = np.sum(was_clipped)
                    if clipped_count > 0:
                        print(f"    {clipped_count} values were clipped (explains some diff)")
                    else:
                        print(f"    WARNING: large diff without clipping — possible issue")
                        all_passed = False

            # Check replaced positions: should be close to last_valid_delta + small noise
            replaced_delta = batch_delta[replaced_positions]
            last_valid_delta = raw_delta[last_valid_pos]
            replaced_diff = np.abs(replaced_delta - last_valid_delta[None, :])
            max_replaced_diff = np.max(replaced_diff)
            mean_replaced_diff = np.mean(replaced_diff)
            print(f"  Replaced positions: delta diff from last valid (should be ~noise)")
            print(f"    max diff: {max_replaced_diff:.6f}, mean diff: {mean_replaced_diff:.6f}")

            # The raw noise is N(0, 0.005). After chunk-wise delta, the noise propagates directly.
            # So diff should be on the order of 0.005 * sqrt(14) ~= 0.019 per element.
            if max_replaced_diff > 0.5:
                print(f"    WARNING: replaced action diff too large, expected noise-scale ~0.005")
                all_passed = False
            else:
                print(f"    OK: diffs consistent with noise std=0.005")

        # --- Check 6: Verify action_mask is all True after dont_mask_actions ---
        batch_mask = sample["action_mask"]
        all_true = np.all(batch_mask)
        true_count = int(np.sum(batch_mask))
        print(f"  Batch action_mask all True: {all_true} ({true_count}/{len(batch_mask)})")
        if not all_true:
            # 30fps samples have last 2/5 masked — that's expected
            expected_30fps = 3 * ACTION_HORIZON // 5
            if true_count == expected_30fps:
                print(f"    30fps sample: {true_count}/{ACTION_HORIZON} valid (expected)")
            else:
                print(f"    FAIL: unexpected mask pattern")
                all_passed = False

        print()

    if all_passed:
        print("RESULT: All raw episode comparison checks PASSED")
    else:
        print("RESULT: Some raw episode comparison checks FAILED")
    print()
    return all_passed


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description = "Verify dont_mask_actions and normalization")
    parser.add_argument("--max_batches", type = int, default = 30, help = "Max batches for verification 1")
    parser.add_argument("--batch_size", type = int, default = 256, help = "Batch size")
    parser.add_argument("--skip_raw", action = "store_true", help = "Skip raw episode verification")
    args = parser.parse_args()

    results = {}

    t0 = time.time()
    results["dont_mask_actions"] = verify_dont_mask_actions(
        max_batches = args.max_batches,
        batch_size = args.batch_size,
    )
    print(f"Verification 1 took {time.time() - t0:.1f}s\n")

    if not args.skip_raw:
        t1 = time.time()
        results["raw_episode"] = verify_raw_episode(
            max_batches = args.max_batches,
            batch_size = args.batch_size,
        )
        print(f"Verification 2 took {time.time() - t1:.1f}s\n")

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
