"""Visualize latent embeddings from PaliGemma-based value function models.

Projects CLS token embeddings (last hidden state before the value head) into 2D
via UMAP and overlays multiple models on the same plot for comparison.

UMAP is fit once over ALL embeddings (all models, all episodes), then per-episode
(or per-subtask) subsets are plotted from the global projection.

Usage:
    python scripts/visualize_latent_embeddings.py \
        --model robocoin_bimanual_paligemma_q_sarsa:gs://bucket/checkpoints/.../exp1 \
        --model robocoin_bimanual_paligemma_q_sarsa:gs://bucket/checkpoints/.../real_hang_finetune_300 \
        --episodes /path/to/traj_270.pkl,/path/to/traj_0.pkl \
        --output-dir outputs/latent_viz \
        [--split-subtasks] \
        [--batch-size 64]
"""

import argparse
import importlib
import logging
import os
import pickle

import flax.nnx as nnx
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import umap

from openpi.robocoin_utils.utils import extract_embeddings
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding

logger = logging.getLogger()

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Visualize latent embeddings from value function models.")
    parser.add_argument(
        "--model",
        action = "append",
        required = True,
        metavar = "CONFIG_NAME:CHECKPOINT_PATH",
        help = "Model specification as config_name:checkpoint_path. Can be repeated.",
    )
    parser.add_argument(
        "--episodes",
        required = True,
        help = "Comma-separated list of paths to traj_*.pkl files.",
    )
    parser.add_argument("--output-dir", required = True, help = "Directory for output plots.")
    parser.add_argument("--split-subtasks", action = "store_true", help = "Produce separate plots per subtask segment.")
    parser.add_argument("--batch-size", type = int, default = 64, help = "Batch size for embedding extraction.")
    return parser.parse_args()


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse 'config_name:checkpoint_path' into (config_name, checkpoint_path)."""
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Model spec must be config_name:checkpoint_path, got: {spec}")
    return parts[0], parts[1]


def get_model_display_name(config_name: str, checkpoint_path: str) -> str:
    """Derive a display name from config name and checkpoint path.

    If the checkpoint path ends with a fine-tune subdirectory (e.g. .../real_hang_finetune_300),
    append it to the config name for disambiguation.
    """
    tail = checkpoint_path.rstrip("/").rsplit("/", 1)[-1]
    if tail.isdigit() or tail == config_name:
        return config_name
    return f"{config_name}/{tail}"


def load_cached_episodes(pkl_paths: list[str]) -> dict[int, list[dict]]:
    """Load cached trajectory pickle files from explicit paths."""
    traj_frames: dict[int, list[dict]] = {}
    for path in pkl_paths:
        filename = os.path.basename(path)
        traj_idx = int(filename.replace("traj_", "").replace(".pkl", ""))
        with open(path, "rb") as f:
            traj_frames[traj_idx] = pickle.load(f)
        logger.info(f"Loaded traj_{traj_idx} ({len(traj_frames[traj_idx])} frames) from {path}")
    logger.info(f"Loaded {len(traj_frames)} episodes total")
    return traj_frames


def compute_param_norm(model: nnx.Module) -> float:
    """Compute param_norm the same way as train_value_function.py."""
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            nnx.Not(nnx_utils.PathRegex(".*target_(network|head)/.*")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    return float(jax.device_get(optax.global_norm(kernel_params)))


def load_model_and_extract(
    config_name: str,
    checkpoint_path: str,
    traj_frames: dict[int, list[dict]],
    batch_size: int,
) -> dict[str, list[np.ndarray]]:
    """Load a single model, extract embeddings from all episodes, then delete the model."""
    spec = importlib.util.spec_from_file_location(
        "train_value_function",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_value_function.py"),
    )
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    init_train_state = train_module.init_train_state

    config = _config.get_config(config_name)

    init_rng = jax.random.PRNGKey(86)
    mesh = sharding.make_mesh(config.fsdp_devices)

    train_state_shape, _ = init_train_state(config, init_rng, mesh, resume = True)

    mngr, _ = _checkpoints.initialize_checkpoint_dir(
        checkpoint_path, keep_period = None, overwrite = False, resume = True
    )
    train_state = _checkpoints.restore_state(mngr, train_state_shape, data_loader = None)

    critic_state = train_state.critic
    params = critic_state.ema_params if critic_state.ema_params is not None else critic_state.params
    model = nnx.merge(critic_state.model_def, params)

    param_norm = compute_param_norm(model)
    logger.info(f"param_norm for {config_name} ({checkpoint_path}): {param_norm:.4f}")

    network = model.network
    action_conditioned = network.action_conditioned

    all_frames = []
    ep_keys = set()
    for traj_idx in sorted(traj_frames.keys()):
        frames = traj_frames[traj_idx]
        ep_key = str(traj_idx)
        ep_keys.add(ep_key)
        for frame_idx, frame in enumerate(frames):
            all_frames.append((ep_key, frame_idx, frame))

    logger.info(f"Extracting embeddings for {config_name} ({len(all_frames)} frames across {len(ep_keys)} episodes)")
    embeddings = extract_embeddings(network, all_frames, ep_keys, action_conditioned, batch_size)

    del model, network, train_state, critic_state, params, mngr
    jax.clear_caches()

    return embeddings


def get_subtask_segments(frames: list[dict]) -> list[tuple[str, int, int]]:
    """Extract subtask segments as (text, start_idx, end_idx) tuples."""
    key = "subtask_1_text"
    if key not in frames[0]:
        return [("full_episode", 0, len(frames))]

    segments = []
    prev = None
    start = 0
    for i, f in enumerate(frames):
        st = f[key]
        if hasattr(st, "item"):
            st = st.item()
        if st != prev:
            if prev is not None and prev.lower() not in ("null", "static", "abnormal"):
                segments.append((prev, start, i))
            start = i
            prev = st
    if prev is not None and prev.lower() not in ("null", "static", "abnormal"):
        segments.append((prev, start, len(frames)))

    if not segments:
        segments = [("full_episode", 0, len(frames))]
    return segments


def get_intersecting_subtasks(traj_frames: dict[int, list[dict]]) -> list[str]:
    """Return subtask texts that appear in every loaded episode."""
    common_subtasks: set[str] | None = None
    for frames in traj_frames.values():
        subtasks = {seg_text for seg_text, _, _ in get_subtask_segments(frames) if seg_text != "full_episode"}
        if common_subtasks is None:
            common_subtasks = subtasks
        else:
            common_subtasks &= subtasks

    return sorted(common_subtasks or [])


def get_segment_frame_indices(
    segments: list[tuple[str, int, int]],
    allowed_subtasks: set[str],
) -> np.ndarray:
    """Return concatenated frame indices for all segments whose text is allowed."""
    selected_ranges = [
        np.arange(seg_start, seg_end)
        for seg_text, seg_start, seg_end in segments
        if seg_text in allowed_subtasks
    ]
    if not selected_ranges:
        return np.array([], dtype = int)
    return np.concatenate(selected_ranges)


def sanitize_subtask_name(subtask_text: str, max_len: int = 40) -> str:
    """Return a filename-safe version of a subtask name."""
    return subtask_text[:max_len].replace(" ", "_").replace("/", "-")


def plot_episode(
    projected: np.ndarray,
    index_map: dict[str, dict[str, np.ndarray]],
    model_names: list[str],
    ep_key: str,
    title: str,
    output_path: str,
    frame_range: tuple[int, int] | None = None,
    frame_indices: np.ndarray | None = None,
):
    """Plot a single episode (or subtask slice) from pre-computed global UMAP projection.

    Args:
        projected: Global UMAP projection [N_total, 2].
        index_map: Nested dict model_name -> ep_key -> array of global indices.
        model_names: Ordered list of model names.
        ep_key: Episode key to plot.
        title: Plot title.
        output_path: Path to save the figure.
        frame_range: Optional (start, end) to slice within the episode's indices.
        frame_indices: Optional explicit frame indices to keep from the episode.
    """
    fig, ax = plt.subplots(figsize = (10, 8))

    for i, name in enumerate(model_names):
        if ep_key not in index_map[name]:
            continue
        indices = index_map[name][ep_key]
        if frame_indices is not None:
            indices = indices[frame_indices]
        elif frame_range is not None:
            indices = indices[frame_range[0] : frame_range[1]]
        if len(indices) == 0:
            continue

        pts = projected[indices]
        color = COLORS[i % len(COLORS)]
        ax.scatter(pts[:, 0], pts[:, 1], c = color, label = name, alpha = 0.6, s = 15)

        if len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], c = color, alpha = 0.2, linewidth = 0.8)
            ax.scatter(pts[0, 0], pts[0, 1], c = color, marker = "^", s = 80, edgecolors = "black", zorder = 5)
            ax.scatter(pts[-1, 0], pts[-1, 1], c = color, marker = "s", s = 80, edgecolors = "black", zorder = 5)

    ax.set_title(title, fontsize = 12)
    ax.legend(fontsize = 9, loc = "best")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    fig.tight_layout()
    fig.savefig(output_path, dpi = 150)
    plt.close(fig)
    logger.info(f"Saved plot: {output_path}")


def plot_combined_episodes(
    projected: np.ndarray,
    index_map: dict[str, dict[str, np.ndarray]],
    model_names: list[str],
    episode_keys: list[str],
    title: str,
    output_path: str,
    per_episode_frame_indices: dict[str, np.ndarray] | None = None,
):
    """Plot multiple episodes together from the pre-computed global UMAP projection."""
    fig, ax = plt.subplots(figsize = (10, 8))

    for i, name in enumerate(model_names):
        color = COLORS[i % len(COLORS)]
        label_added = False
        for ep_key in episode_keys:
            if ep_key not in index_map[name]:
                continue

            indices = index_map[name][ep_key]
            if per_episode_frame_indices is not None and ep_key in per_episode_frame_indices:
                indices = indices[per_episode_frame_indices[ep_key]]
            if len(indices) == 0:
                continue

            pts = projected[indices]
            label = name if not label_added else None
            ax.scatter(pts[:, 0], pts[:, 1], c = color, label = label, alpha = 0.6, s = 15)

            if len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], c = color, alpha = 0.2, linewidth = 0.8)
                ax.scatter(pts[0, 0], pts[0, 1], c = color, marker = "^", s = 80, edgecolors = "black", zorder = 5)
                ax.scatter(pts[-1, 0], pts[-1, 1], c = color, marker = "s", s = 80, edgecolors = "black", zorder = 5)

            label_added = True

    ax.set_title(title, fontsize = 12)
    ax.legend(fontsize = 9, loc = "best")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    fig.tight_layout()
    fig.savefig(output_path, dpi = 150)
    plt.close(fig)
    logger.info(f"Saved plot: {output_path}")


def main():
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s %(levelname)s %(name)s: %(message)s", force = True)

    platform = os.environ.get("PLATFORM", "gpu")
    if platform == "tpu":
        jax.distributed.initialize()
        logger.info(f"Initialized JAX distributed: process {jax.process_index()} of {jax.process_count()}")

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok = True)

    model_specs = [parse_model_spec(s) for s in args.model]
    logger.info(f"Models to compare: {[s[0] for s in model_specs]}")

    pkl_paths = [p.strip() for p in args.episodes.split(",")]
    traj_frames = load_cached_episodes(pkl_paths)
    if not traj_frames:
        raise ValueError(f"No episodes loaded from: {args.episodes}")

    per_model_embeddings: dict[str, dict[str, list[np.ndarray]]] = {}
    model_names: list[str] = []

    for config_name, checkpoint_path in model_specs:
        display_name = get_model_display_name(config_name, checkpoint_path)
        model_names.append(display_name)

        logger.info(f"Loading model: {display_name} from {checkpoint_path}")
        embeddings = load_model_and_extract(config_name, checkpoint_path, traj_frames, args.batch_size)
        per_model_embeddings[display_name] = embeddings
        logger.info(f"Done with {display_name}")

    if jax.process_index() == 0:
        intersecting_subtasks = get_intersecting_subtasks(traj_frames)
        if intersecting_subtasks:
            logger.info(f"Intersecting subtasks across episodes: {intersecting_subtasks}")
        else:
            logger.info("No intersecting subtasks found across episodes")

        # Stack per-episode embeddings into arrays
        ep_keys_sorted = [str(k) for k in sorted(traj_frames.keys())]
        per_model_stacked: dict[str, dict[str, np.ndarray]] = {}
        for name in model_names:
            per_model_stacked[name] = {}
            for ep_key in ep_keys_sorted:
                ep_embeds = per_model_embeddings[name].get(ep_key, [])
                if ep_embeds:
                    per_model_stacked[name][ep_key] = np.stack(ep_embeds, axis = 0)

        # Build global embedding matrix and track indices per (model, episode)
        all_embeds = []
        index_map: dict[str, dict[str, np.ndarray]] = {name: {} for name in model_names}
        global_offset = 0

        for name in model_names:
            for ep_key in ep_keys_sorted:
                if ep_key not in per_model_stacked[name]:
                    continue
                ep_arr = per_model_stacked[name][ep_key]
                n = len(ep_arr)
                index_map[name][ep_key] = np.arange(global_offset, global_offset + n)
                all_embeds.append(ep_arr)
                global_offset += n

        global_embeds = np.concatenate(all_embeds, axis = 0)
        logger.info(f"Fitting UMAP on {len(global_embeds)} total embeddings ({len(model_names)} models x {len(ep_keys_sorted)} episodes)")

        n_neighbors = min(30, len(global_embeds) - 1)
        reducer = umap.UMAP(n_neighbors = n_neighbors, min_dist = 0.05, metric = "cosine", random_state = 86)
        projected = reducer.fit_transform(global_embeds)

        if intersecting_subtasks:
            for subtask_text in intersecting_subtasks:
                intersecting_frame_indices = {}
                for traj_idx in sorted(traj_frames.keys()):
                    ep_key = str(traj_idx)
                    segments = get_subtask_segments(traj_frames[traj_idx])
                    indices = get_segment_frame_indices(segments, {subtask_text})
                    if len(indices) > 0:
                        intersecting_frame_indices[ep_key] = indices

                if intersecting_frame_indices:
                    safe_text = sanitize_subtask_name(subtask_text)
                    output_path = os.path.join(
                        args.output_dir,
                        f"intersecting_subtask_{safe_text}.png",
                    )
                    plot_combined_episodes(
                        projected,
                        index_map,
                        model_names,
                        ep_keys_sorted,
                        title = f"Intersecting Subtask: {subtask_text}",
                        output_path = output_path,
                        per_episode_frame_indices = intersecting_frame_indices,
                    )

        # Plot per episode (or per subtask)
        for traj_idx in sorted(traj_frames.keys()):
            ep_key = str(traj_idx)
            frames = traj_frames[traj_idx]
            segments = get_subtask_segments(frames)

            output_path = os.path.join(args.output_dir, f"episode_{traj_idx}.png")
            plot_episode(
                projected, index_map, model_names, ep_key,
                title = f"Episode {traj_idx} — Latent Embeddings",
                output_path = output_path,
            )

            if args.split_subtasks:
                for seg_idx, (seg_text, seg_start, seg_end) in enumerate(segments):
                    safe_text = sanitize_subtask_name(seg_text)
                    output_path = os.path.join(
                        args.output_dir, f"episode_{traj_idx}_seg{seg_idx}_{safe_text}.png"
                    )
                    plot_episode(
                        projected, index_map, model_names, ep_key,
                        title = f"Episode {traj_idx} — Segment {seg_idx}: {seg_text}",
                        output_path = output_path,
                        frame_range = (seg_start, seg_end),
                    )

        logger.info(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
