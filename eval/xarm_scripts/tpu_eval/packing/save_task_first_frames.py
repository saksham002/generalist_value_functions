"""Save the first-frame scene image for every task in packing_tasks.json.

Each task entry carries a `repo_id` (the data round, e.g. "xarm_baseline_round2") and a
`subtasks` list of "Pack <object> into the <size> box" steps. The recorded episodes live under
`<data_root>/<repo_id>/<episode_dir>/episode_*.npz`, where each npz stores `obses` (per-step
observation dicts) and the matching `audio_transcriptions.json` holds the spoken subtask
commands for that episode.

The original `global_chunk_index -> episode_dir` mapping (from the now-absent chunks.json) is
not recoverable, so this script instead matches each task to its episode by CONTENT: the set of
(object, box) pairs parsed from the task's subtasks is matched against the pack commands in each
episode's audio_transcriptions.json. The (object, box) multiset uniquely identifies an episode
within a round, and the match confidence is printed so each result can be spot-checked.

For the matched episode, the first frame of the chosen camera (default "right/top", the overhead
scene view) is JPEG-decoded from obses[0]["images"] and written as a PNG.

Usage:
    python eval/xarm_scripts/tpu_eval/packing/save_task_first_frames.py \
        --data-root "/media/huzheyuan/data0/huzheyuan_folder_backup/dual_xarms/dual_xarms_sim/adversarial_project/all_data" \
        --output-dir eval/xarm_scripts/tpu_eval/packing/task_first_frames
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import cv2
import numpy as np

_DEFAULT_DATA_ROOT = (
    "/media/huzheyuan/data0/huzheyuan_folder_backup/dual_xarms/dual_xarms_sim/"
    "adversarial_project/all_data"
)
_CATEGORIES = ("small_medium", "small_large", "medium_large")
_PACK_RE = re.compile(r"pack\s+(.*?)\s+into\s+(?:the\s+)?(small|medium|large)\s+box", re.IGNORECASE)


def _normalize_object(name: str) -> str:
    """Lowercase, drop a leading article and punctuation, collapse whitespace."""
    name = name.lower().replace(".", "").strip()
    name = re.sub(r"^(the|a|an)\s+", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def pack_pairs(strings: list[str]) -> set[str]:
    """Parse a list of 'Pack X into the Y box' strings into a set of 'object|box' keys."""
    pairs: set[str] = set()
    for s in strings:
        m = _PACK_RE.search(s)
        if m:
            pairs.add(f"{_normalize_object(m.group(1))}|{m.group(2).lower()}")
    return pairs


def episode_pack_pairs(episode_dir: str) -> set[str]:
    """Pack pairs spoken during an episode, from its audio_transcriptions.json."""
    transcript_path = os.path.join(episode_dir, "audio_transcriptions.json")
    if not os.path.exists(transcript_path):
        return set()
    with open(transcript_path) as f:
        entries = json.load(f)
    return pack_pairs([e.get("transcription", "") for e in entries])


def match_episode(round_dir: str, want: set[str]) -> tuple[str | None, float, str | None, float]:
    """Find the episode dir whose pack pairs best match `want`.

    Returns (best_dir, best_score, second_dir, second_score) where score is the fraction of
    `want` pairs found in the episode (1.0 == exact subset match).
    """
    scored: list[tuple[float, str]] = []
    for episode_dir in sorted(glob.glob(os.path.join(round_dir, "*", ""))):
        have = episode_pack_pairs(episode_dir)
        if not have:
            continue
        score = len(want & have) / max(len(want), 1)
        scored.append((score, episode_dir.rstrip("/")))
    if not scored:
        return None, 0.0, None, 0.0
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_dir = scored[0]
    second_dir, second_score = (scored[1][1], scored[1][0]) if len(scored) > 1 else (None, 0.0)
    return best_dir, best_score, second_dir, second_score


def save_first_frame(episode_dir: str, camera: str, out_path: str) -> tuple[int, int]:
    """Decode and write the first-frame image for `camera`. Returns (height, width)."""
    npz_files = glob.glob(os.path.join(episode_dir, "episode_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No episode_*.npz in {episode_dir}")
    with np.load(npz_files[0], allow_pickle=True) as data:
        first_obs = data["obses"][0]
    images = first_obs["images"]
    if camera not in images:
        raise KeyError(f"Camera {camera!r} not in obs images {list(images.keys())} ({episode_dir})")
    # Frames are stored as JPEG-encoded byte arrays; imdecode returns BGR, which imwrite expects.
    jpeg_bytes = np.asarray(images[camera], dtype=np.uint8)
    frame_bgr = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError(f"Failed to JPEG-decode camera {camera!r} in {episode_dir}")
    cv2.imwrite(out_path, frame_bgr)
    return frame_bgr.shape[0], frame_bgr.shape[1]


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-file", default=os.path.join(script_dir, "packing_tasks.json"))
    parser.add_argument("--data-root", default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=os.path.join(script_dir, "task_first_frames"))
    parser.add_argument("--camera", default="right/top", help="Camera key to extract (default: right/top, the overhead scene view).")
    parser.add_argument(
        "--min-score", type=float, default=1.0,
        help="Minimum match score (fraction of task subtasks found) required to save a frame. "
             "Tasks below this are reported and skipped.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.tasks_file) as f:
        tasks_data = json.load(f)

    global_index = 0
    saved, skipped = 0, 0
    for category in _CATEGORIES:
        for entry in tasks_data.get(category, []):
            chunk = entry.get("global_chunk_index")
            repo_id = entry["repo_id"]
            want = pack_pairs(entry.get("subtasks", []))
            round_dir = os.path.join(args.data_root, repo_id)

            label = f"[{global_index:2d}] {category:>12} chunk={chunk} {repo_id}"
            global_index += 1

            if not want:
                print(f"{label}: SKIP — no parseable subtasks in task entry.")
                skipped += 1
                continue
            if not os.path.isdir(round_dir):
                print(f"{label}: SKIP — round dir not found: {round_dir}")
                skipped += 1
                continue

            best_dir, best_score, second_dir, second_score = match_episode(round_dir, want)
            if best_dir is None or best_score < args.min_score:
                print(
                    f"{label}: SKIP — best match {os.path.basename(best_dir) if best_dir else None} "
                    f"score={best_score:.2f} < min_score={args.min_score:.2f}"
                )
                skipped += 1
                continue

            episode_name = os.path.basename(best_dir)
            ambiguous = second_dir is not None and second_score >= best_score
            cam_tag = args.camera.replace("/", "_")
            out_path = os.path.join(args.output_dir, f"{repo_id}_chunk{chunk}_{episode_name}_{cam_tag}.png")
            try:
                h, w = save_first_frame(best_dir, args.camera, out_path)
            except (FileNotFoundError, KeyError, ValueError) as e:
                print(f"{label}: ERROR saving frame from {episode_name}: {e}")
                skipped += 1
                continue

            warn = "  ** AMBIGUOUS (tie with %s @ %.2f)" % (os.path.basename(second_dir), second_score) if ambiguous else ""
            print(f"{label}: {episode_name} score={best_score:.2f} -> {os.path.basename(out_path)} ({w}x{h}){warn}")
            saved += 1

    print(f"\nDone: {saved} saved, {skipped} skipped. Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
