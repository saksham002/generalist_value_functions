#!/usr/bin/env python3
"""
Test the iteration speed of the DLIMP DataLoader for the RoboCOIN TFDS dataset.
"""

import argparse
import time
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import jax
import pdb
from openpi.training.config import RoboCOINDataConfig, TrainConfig
from openpi.training import data_loader as _data_loader
from openpi.training.robocoin_data_loader import RoboCOINDataLoaderConfig, create_robocoin_data_loader

from openpi.value_functions import value_function as _value_function
import openpi.value_functions.networks.paligemma as _paligemma_network
from openpi.value_functions import heads as _heads
from openpi.models.model import IMAGE_KEYS, Observation
from openpi.models.tokenizer import create_tokenizer
from openpi.models import pi0_config

# ---------- Configuration ----------
DATA_DIR = "gs://saksham-euw4/robocoin_bimanual/"
DATASET_NAME = "robocoin:1.0.0"
SPLIT = "train"
# Directory for saving plots
PLOTS_DIR = Path(__file__).parent / "plots" / "images"
PLOTS_DIR.mkdir(parents = True, exist_ok = True)
VIDEOS_DIR = Path(__file__).parent / "plots" / "videos"
VIDEOS_DIR.mkdir(parents = True, exist_ok = True)

DEFAULT_REPO_ID = "RoboCOIN/Cobot_Magic_cut_banana"

def save_episode_video(
    frames: List[Dict[str, Any]],
    repo_id: str,
    episode_idx: int,
    fps: int,
    videos_dir: Path = VIDEOS_DIR,
):
    """Save all frames of an episode as an mp4, showing all three cameras side by side."""
    import cv2

    camera_keys = ["left_wrist_0_rgb", "right_wrist_0_rgb", "base_0_rgb"]
    repo_id_safe = repo_id.replace("/", "_")
    save_path = videos_dir / f"{repo_id_safe}_{episode_idx}.mp4"

    rendered_frames = []
    for frame in frames:
        fig, axes = plt.subplots(1, len(camera_keys), figsize=(8 * len(camera_keys), 8))
        for ax, cam_key in zip(axes, camera_keys):
            ax.imshow(np.asarray(frame["image"][cam_key]))
            ax.axis("off")
            ax.set_title(cam_key)
        plt.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        rendered_frames.append(buf.copy())
        plt.close(fig)

    h, w = rendered_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for buf in rendered_frames:
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"  Saved video ({len(rendered_frames)} frames @ {fps} fps) to {save_path}")


def plot_batch_images(batch: Dict[str, Any], index: int, plots_dir: Path = PLOTS_DIR, split: str = "train"):
    """Plot and save images from a specific example in the batch.

    Only plots images that have non-zero values. For val split, also plots
    all cameras from mirror_image (if present).

    Args:
        batch: Batch dictionary from the iterator (standard format)
        index: Index of the example within the batch to plot
        plots_dir: Directory to save plots
        split: Dataset split ("train" or "val")
    """
    def to_numpy(x):
        if hasattr(x, "device_buffer"):
            return np.array(x)
        return x

    if "image" not in batch:
        print("  No 'image' key in batch.")
        return

    for key, image_batch in batch["image"].items():
        image_batch = to_numpy(image_batch)
        image = image_batch[index] if image_batch.ndim > 0 else image_batch

        if image.size == 0 or np.prod(image.shape) == 0 or np.all(image == 0):
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f'{key}')
        save_path = plots_dir / f'{key}.png'
        fig.savefig(save_path, bbox_inches = 'tight', dpi = 100)
        plt.close(fig)
        print(f"  Saved image to {save_path}")

    if split == "val" and "mirror_image" in batch:
        for key, image_batch in batch["mirror_image"].items():
            image_batch = to_numpy(image_batch)
            image = image_batch[index] if image_batch.ndim > 0 else image_batch

            if image.size == 0 or np.prod(image.shape) == 0 or np.all(image == 0):
                continue

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(f"mirror_{key}")
            save_path = plots_dir / f"mirror_{key}.png"
            fig.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close(fig)
            print(f"  Saved mirror image to {save_path}")


def _print_dict_structure(d: Dict[str, Any], prefix: str = "  ") -> None:
    """Print keys, shapes, and dtypes of a nested dict."""
    for key in sorted(d.keys()):
        val = d[key]
        if isinstance(val, dict):
            print(f"{prefix}{key}: (dict with {len(val)} keys)")
            _print_dict_structure(val, prefix = prefix + "  ")
        elif hasattr(val, "shape"):
            print(f"{prefix}{key}: shape={val.shape}, dtype={val.dtype}")
        elif isinstance(val, (list, tuple)):
            print(f"{prefix}{key}: {type(val).__name__}, len={len(val)}")
        else:
            print(f"{prefix}{key}: {type(val).__name__} = {val}")


def _print_observation_structure(obs: Observation) -> None:
    """Print the fields of an Observation dataclass."""
    print("  Observation fields:")
    for field_name in ["images", "image_masks", "state", "tokenized_prompt", "tokenized_prompt_mask",
                       "action_mask", "loss_mask"]:
        val = getattr(obs, field_name, None)
        if val is None:
            print(f"    {field_name}: None")
        elif isinstance(val, dict):
            print(f"    {field_name}: (dict with {len(val)} keys)")
            for k, v in val.items():
                if hasattr(v, "shape"):
                    print(f"      {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"      {k}: {type(v).__name__}")
        elif hasattr(val, "shape"):
            print(f"    {field_name}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"    {field_name}: {type(val).__name__}")


def iterate_dataset(
    iterator,
    batch_size: int = 512,
    max_batches: Optional[int] = None,
    plot_images: bool = False,
    save_video: bool = False,
    repo_id: str = DEFAULT_REPO_ID,
    split: str = "train",
    mode: str = "critic",
    use_quantile_norm: bool = False,
):
    """Iterate through the dataset and measure throughput."""
    print("\n==================== Iterating Through Dataset ====================")
    print(f"Mode: {mode} | Batch size: {batch_size} | Max batches: {max_batches or 'unlimited'}")

    total_samples = 0
    total_batches = 0
    batch_times = []
    start_time = time.time()
    batch_start = None

    clip_bound = 1.0 if use_quantile_norm else 5.0
    state_at_clip = []
    actions_at_clip = []
    loss_mask_sum = 0
    loss_mask_count = 0

    for batch_idx, raw_batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            print(f"\nReached max_batches limit ({max_batches}). Stopping.")
            break

        # Unpack based on mode: critic yields dict, pi05 yields (Observation, Actions)
        if mode == "pi05":
            assert isinstance(raw_batch, tuple) and len(raw_batch) == 2, (
                f"pi05 mode expected (Observation, Actions) tuple, got {type(raw_batch)}"
            )
            obs, actions = raw_batch
            assert isinstance(obs, Observation), f"Expected Observation, got {type(obs)}"
            current_batch_size = actions.shape[0]
        else:
            assert isinstance(raw_batch, dict), f"critic mode expected dict, got {type(raw_batch)}"
            batch = raw_batch
            current_batch_size = next(
                v.shape[0] for v in jax.tree.leaves(batch) if hasattr(v, "shape") and len(v.shape) > 0
            )

        # Track fraction of values at the clip bound (exclude zero-padding)
        if mode == "pi05":
            s = np.asarray(obs.state)[..., :14]
            a = np.asarray(actions)[..., 14:28]
        else:
            s = np.asarray(batch["state"])
            a = np.asarray(batch["actions"])
        state_at_clip.append(np.mean(np.abs(s) >= clip_bound - 1e-6))
        actions_at_clip.append(np.mean(np.abs(a) >= clip_bound - 1e-6))

        if mode == "pi05":
            if obs.loss_mask is not None:
                lm = np.asarray(obs.loss_mask).astype(np.float32)
                loss_mask_sum += lm.sum()
                loss_mask_count += lm.size
        else:
            if "loss_mask" in batch:
                lm = np.asarray(batch["loss_mask"]).astype(np.float32)
                loss_mask_sum += lm.sum()
                loss_mask_count += lm.size

        total_samples += current_batch_size
        total_batches += 1

        batch_end = time.time()
        batch_times.append(batch_end - batch_start if batch_start is not None else 0)

        # Detailed diagnostics on the first batch
        if batch_idx == 0:
            print(f"\n--- First batch diagnostics (mode={mode}) ---")
            print(f"  Actual batch size: {current_batch_size}")

            if mode == "pi05":
                _print_observation_structure(obs)
                print(f"  Actions: shape={actions.shape}, dtype={actions.dtype}")
                print(f"  Actions range: [{np.asarray(actions).min():.4f}, {np.asarray(actions).max():.4f}]")

                # Verify PI0.5-specific expectations
                state = np.asarray(obs.state)
                print(f"  State shape: {state.shape} (expect [B, 32] with offset=14)")
                print(f"  State[:, :14] range: [{state[:, :14].min():.4f}, {state[:, :14].max():.4f}]")
                print(f"  State[:, 14:] (padding): all zeros = {np.allclose(state[:, 14:], 0)}")

                tok = np.asarray(obs.tokenized_prompt)
                tok_mask = np.asarray(obs.tokenized_prompt_mask)
                print(f"  Tokenized prompt: shape={tok.shape}, non-pad tokens (sample 0): {tok_mask[0].sum()}")

                if obs.action_mask is not None:
                    am = np.asarray(obs.action_mask)
                    print(f"  action_mask: shape={am.shape}, true count (sample 0)={am[0].sum()}")

                if obs.loss_mask is not None:
                    lm = np.asarray(obs.loss_mask)
                    print(f"  loss_mask: shape={lm.shape}, true frac={lm.mean():.3f}")

            else:
                print(f"  Batch keys: {sorted(batch.keys())}")
                _print_dict_structure(batch)

                # Critic-specific checks
                state = np.asarray(batch["state"])
                print(f"\n  State range: [{state.min():.4f}, {state.max():.4f}]")
                actions_arr = np.asarray(batch["actions"])
                print(f"  Actions range: [{actions_arr.min():.4f}, {actions_arr.max():.4f}]")

                if "tokenized_prompt" in batch:
                    tok = np.asarray(batch["tokenized_prompt"])
                    tok_mask = np.asarray(batch["tokenized_prompt_mask"])
                    print(f"  Tokenized prompt: shape={tok.shape}, non-pad (sample 0): {tok_mask[0].sum()}")

                if "loss_mask" in batch:
                    lm = np.asarray(batch["loss_mask"])
                    print(f"  loss_mask: true frac={lm.mean():.3f}")

                if "termination" in batch:
                    term = np.asarray(batch["termination"])
                    print(f"  Terminations in batch: {term.sum()}")

            print("--- End first batch diagnostics ---\n")

        # Progress logging
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
            avg_batch_time = np.mean(batch_times) if batch_times else 0
            print(
                f"  Batch {batch_idx + 1}: "
                f"samples={total_samples}, "
                f"elapsed={elapsed:.1f}s, "
                f"throughput={samples_per_sec:.1f} samples/s, "
                f"avg_batch_time={avg_batch_time:.3f}s"
            )

        batch_start = time.time()

    elapsed_time = time.time() - start_time
    samples_per_sec = total_samples / elapsed_time if elapsed_time > 0 else 0

    print("\n==================== Iteration Complete ====================")
    print(f"Total batches: {total_batches}")
    print(f"Total samples: {total_samples}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Throughput: {samples_per_sec:.2f} samples/s")
    if state_at_clip:
        print(f"Clip bound: {clip_bound}")
        print(f"State at clip (mean across batches): {np.mean(state_at_clip):.4f}")
        print(f"Actions at clip (mean across batches): {np.mean(actions_at_clip):.4f}")
    if loss_mask_count > 0:
        print(f"loss_mask valid fraction (across all batches): {loss_mask_sum / loss_mask_count:.4f}")
    print("=============================================================")

    return {
        "total_batches": total_batches,
        "total_samples": total_samples,
        "elapsed_time": elapsed_time,
        "samples_per_sec": samples_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test DLIMP DataLoader for RoboCOIN TFDS dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for data loading (default: 256)"
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=75,
        help="Maximum number of batches to iterate (default: None for unlimited)"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Whether to shuffle the data"
    )
    parser.add_argument(
        "--local_shuffle_buffer_size",
        type=int,
        default=50000,
        help="Size of shuffle buffer (default: 10000)"
    )
    parser.add_argument(
        "--prefetch_buffer_size",
        type=int,
        default=4,
        help="Number of batches to prefetch (default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=86,
        help="Random seed for shuffling (default: 86)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Repo ID to target for --plot_images / --save_video (default: {DEFAULT_REPO_ID})"
    )
    parser.add_argument(
        "--plot_images",
        action="store_true",
        help="Plot and save images from the first sample matching --repo_id"
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save the first complete episode matching --repo_id as an mp4 (overrides --plot_images)"
    )
    parser.add_argument(
        "--use_quantile_norm",
        action="store_true",
        help="Use min-max (quantile) normalization for state instead of z-score"
    )
    parser.add_argument(
        "--norm_stats_path",
        type=str,
        default="gs://saksham-euw4/robocoin_bimanual/norm_stats/embodiment_wise_stats.json",
        help="Path to norm_stats.json file for state normalization"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="critic",
        choices=["critic", "pi05"],
        help="critic: value function config (critic_mode=True), pi05: policy config (critic_mode=False)"
    )
    parser.add_argument(
        "--no_state",
        action="store_true",
        default=False,
        help="Disable discrete state input (sets discrete_state_input=False in Pi0Config)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("DLIMP DataLoader Test for RoboCOIN Dataset (via Training Pipeline)")
    print("=" * 60)
    
    if args.mode == "pi05":
        # PI0.5 policy config (critic_mode=False): tests model_transforms pipeline
        robocoin_data_config = RoboCOINDataConfig(
            tfds_data_dir=DATA_DIR,
            dataset_name=DATASET_NAME,
            discount=0.999,
            local_shuffle_buffer_size=args.local_shuffle_buffer_size,
            norm_stats_path=args.norm_stats_path,
            use_quantile_norm=True,
            use_eef=True,
            td_n=50,
            critic_mode=False,
            use_chunk_wise_delta=True,
        )
        train_config = TrainConfig(
            name="test_robocoin_pi05",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b",
                action_expert_variant="gemma_300m",
                action_dim=32,
                action_horizon=50,
                max_token_len=96,
                pi05=True,
                action_dim_offset=14,
                action_dim_mask=(False,) * 14 + (True,) * 14 + (False,) * 4,
                discrete_state_input=not args.no_state,
            ),
            data=robocoin_data_config,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=0,
            wandb_enabled=False,
            action_horizon=50,
        )
    else:
        # Critic / value function config (critic_mode=True)
        robocoin_data_config = RoboCOINDataConfig(
            tfds_data_dir=DATA_DIR,
            dataset_name=DATASET_NAME,
            discount=0.999,
            local_shuffle_buffer_size=args.local_shuffle_buffer_size,
            norm_stats_path=args.norm_stats_path,
            use_quantile_norm=args.use_quantile_norm,
            use_eef=True,
            td_n=50,
            dont_mask_actions=True,
        )
        train_config = TrainConfig(
            name="test_robocoin_dataloader",
            model=_value_function.SARSAValueFunctionConfig(
                network_config=_paligemma_network.PaliGemmaNetworkConfig(
                    state_dim=14,
                    num_cameras=3,
                    image_size=(224, 224),
                    max_token_len=48,
                    action_dim=14,
                    no_state=True,
                ),
                head_config=_heads.RegressionHeadConfig(),
            ),
            data=robocoin_data_config,
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=0,
            wandb_enabled=False,
            action_horizon=50,
        )

    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Shuffle: {args.shuffle}")
    print(f"  State normalization path: {args.norm_stats_path}")
    print(f"  Use quantile norm: {args.use_quantile_norm}")
    print(f"  Split: {args.split}")
    print(f"  critic_mode: {robocoin_data_config.critic_mode}")
    if args.mode == "pi05":
        print(f"  discrete_state_input: {not args.no_state}")
    print()

    if args.split == "train":
        data_loader = _data_loader.create_data_loader(
            train_config,
            shuffle=args.shuffle,
            num_batches=args.max_batches,
        )
        iterator = iter(data_loader)
    else:
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

        val_loader_config = RoboCOINDataLoaderConfig(
            data_dir=robocoin_data_config.tfds_data_dir,
            dataset_name=robocoin_data_config.dataset_name,
            split="val",
            batch_size=args.batch_size,
            shuffle=False,
            repeat=False,
            seed=args.seed,
            max_cameras=3,
            max_state_dim=14,
            max_action_dim=14,
            image_size=(224, 224),
            discount=robocoin_data_config.discount,
            td_n=50,
            state_norm_stats=data_config.norm_stats,
            use_quantile_norm=data_config.use_quantile_norm,
            use_eef=robocoin_data_config.use_eef,
            action_horizon=train_config.action_horizon,
            dont_mask_actions=True,
        )
        print(f"  Val loader config: {val_loader_config}")

        val_tokenizer = create_tokenizer("paligemma", val_loader_config.max_token_len)
        iterator = create_robocoin_data_loader(val_loader_config, tokenizer = val_tokenizer)

    # Iterate through dataset and measure performance
    iterate_dataset(
        iterator,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        plot_images=args.plot_images,
        save_video=args.save_video,
        repo_id=args.repo_id,
        split=args.split,
        mode=args.mode,
        use_quantile_norm=robocoin_data_config.use_quantile_norm,
    )
    
if __name__ == "__main__":
    main()
