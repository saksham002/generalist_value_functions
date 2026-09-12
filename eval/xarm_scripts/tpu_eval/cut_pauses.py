"""Cut the policy-inference pauses out of an external camera recording of an eval episode.

The eval clients hold the robot still while they wait for the policy server. With
``--args.log-pause-timing`` they write ``episode_<n>_pause_timing.json`` next to the episode
videos and play a sync tone at the first and last step. This script maps every logged pause
onto the camera's timeline, using the tones found in the camera's audio track, and drives one
ffmpeg pass that drops those frames. Because the robot is stationary on both sides of every
pause, the joins read as continuous motion.

Usage:
    uv run eval/xarm_scripts/tpu_eval/cut_pauses.py \
        --args.camera-video /path/to/DSC_0001.MOV \
        --args.timing-json eval/xarm_scripts/tpu_eval/packing/videos/website/episode_3_pause_timing.json

    # No speaker on the client machine: give the camera time (s) at which the robot first moved.
    uv run eval/xarm_scripts/tpu_eval/cut_pauses.py ... --args.sync manual --args.first-motion-time 4.37

Requires ``ffmpeg`` and ``ffprobe`` on PATH. Also writes ``<output>.cuts.csv`` with every cut in
camera time so the edit can be checked or redone by hand.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Literal

import numpy as np
import tyro

_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from eval.helpers import pause_timing as _pause_timing

_AUDIO_SAMPLE_RATE_HZ = 16000
_MOTION_FRAME_WIDTH = 128
_MOTION_FRAME_HEIGHT = 72


@dataclasses.dataclass
class Args:
    camera_video: str
    """Recording from the external camera, covering the whole episode including both sync tones."""

    timing_json: str
    """``episode_<n>_pause_timing.json`` written by the eval client for the same episode."""

    output: str = ""
    """Output mp4. Defaults to ``<camera_video stem>_cut.mp4`` next to the input."""

    sync: Literal["audio", "manual"] = "audio"
    """``audio``: locate the sync tones in the camera's audio track. ``manual``: use
    ``first_motion_time`` as the camera time of the first env step."""

    first_motion_time: float | None = None
    """Camera time (s) at which the robot first moves; required with ``--sync manual``."""

    margin_frames: float = 1.0
    """Frames kept on each side of every pause, so a sync error of up to this many frames never
    removes moving frames. Each frame of margin leaves that much stillness in the output."""

    pause_threshold_s: float | None = None
    """Minimum gap between consecutive steps that counts as a pause. Defaults to the value the
    client logged (two control periods)."""

    refine_offset: bool = True
    """Shift the clock map within ``refine_window_s`` to minimise camera motion inside the pauses.
    Absorbs the tone player's start-up latency; needs a decode of the camera video."""

    refine_window_s: float = 0.15
    """Half-width of the offset search used by ``refine_offset``."""

    motion_check: bool = True
    """Flag pauses whose camera frames contain motion (e.g. a person walking through the shot)."""

    keep_audio: bool = False
    """Keep the camera's audio track (cut in sync with the video). Off by default: the sync
    tones are on it and website videos are usually silent."""

    crf: int = 16
    """libx264 quality; 16 is visually lossless for 1080p/4K footage."""

    dry_run: bool = False
    """Write the cut list and print the ffmpeg command without encoding."""


# =============================================================================
# ffmpeg helpers
# =============================================================================


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"`{name}` not found on PATH; install ffmpeg to use this script.")
    return path


def probe_video(ffprobe: str, path: str) -> tuple[float, float, bool]:
    """Return (fps, duration_s, has_audio) for the first video stream."""
    result = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
        check = True,
        capture_output = True,
        text = True,
    )
    info = json.loads(result.stdout)
    video_streams = [s for s in info["streams"] if s.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError(f"{path}: no video stream found.")
    numerator, denominator = video_streams[0]["r_frame_rate"].split("/")
    fps = float(numerator) / float(denominator)
    duration_s = float(info["format"]["duration"])
    has_audio = any(s.get("codec_type") == "audio" for s in info["streams"])
    return fps, duration_s, has_audio


def read_audio_mono(ffmpeg: str, path: str, sample_rate_hz: int) -> np.ndarray:
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sample_rate_hz), "-f", "s16le", "-"],
        check = True,
        capture_output = True,
    )
    return np.frombuffer(result.stdout, dtype = np.int16).astype(np.float64)


def read_motion_energy(ffmpeg: str, path: str, fps: float) -> np.ndarray:
    """Decode the video to small grayscale frames and return per-frame motion energy."""
    result = subprocess.run(
        [
            ffmpeg, "-v", "error", "-i", path, "-an",
            "-vf", f"fps={fps},scale={_MOTION_FRAME_WIDTH}:{_MOTION_FRAME_HEIGHT},format=gray",
            "-f", "rawvideo", "-",
        ],
        check = True,
        capture_output = True,
    )
    frame_bytes = _MOTION_FRAME_WIDTH * _MOTION_FRAME_HEIGHT
    num_frames = len(result.stdout) // frame_bytes
    frames = np.frombuffer(result.stdout[: num_frames * frame_bytes], dtype = np.uint8)
    frames = frames.reshape(num_frames, _MOTION_FRAME_HEIGHT, _MOTION_FRAME_WIDTH)
    return _pause_timing.frame_motion_energy(frames)


def write_filter_script(path: str, intervals: list[tuple[float, float]], *, with_audio: bool) -> None:
    """Write an ffmpeg filter graph that drops every frame inside any of the intervals."""
    drop = "+".join(f"between(t,{start:.4f},{end:.4f})" for start, end in intervals)
    lines = [f"[0:v]select='not({drop})',setpts=N/FRAME_RATE/TB[v]"]
    if with_audio:
        lines.append(f"[0:a]aselect='not({drop})',asetpts=N/SR/TB[a]")
    with open(path, "w") as f:
        f.write(";\n".join(lines) + "\n")


# =============================================================================
# Main
# =============================================================================


def build_clock_map(args: Args, record: dict, ffmpeg: str, has_audio: bool) -> _pause_timing.ClockMap:
    tone = record["tone"]
    if args.sync == "manual":
        if args.first_motion_time is None:
            raise ValueError("--args.first-motion-time is required with --args.sync manual.")
        first_step_sent = record["steps"]["sent"][0]
        return _pause_timing.ClockMap(wall_reference = first_step_sent, camera_reference = args.first_motion_time)

    if tone["start_time"] is None:
        raise ValueError(
            "The timing log has no sync tone (the client had no `aplay`); use --args.sync manual "
            "with --args.first-motion-time."
        )
    if not has_audio:
        raise ValueError(f"{args.camera_video} has no audio track; use --args.sync manual.")
    audio = read_audio_mono(ffmpeg, args.camera_video, _AUDIO_SAMPLE_RATE_HZ)
    onsets = _pause_timing.detect_tones(audio, _AUDIO_SAMPLE_RATE_HZ, tone["frequency_hz"], tone["duration_s"])
    print(f"Detected {len(onsets)} tone(s) in the camera audio at {[round(t, 3) for t in onsets]} s")
    if not onsets:
        raise ValueError("No sync tone found in the camera audio; check the recording level or use --args.sync manual.")
    if tone["end_time"] is None or len(onsets) < 2:
        if tone["end_time"] is not None:
            print("Only one tone found; using the start tone alone (no drift correction).")
        return _pause_timing.clock_map_from_tones(tone["start_time"], onsets[0], None, None)
    return _pause_timing.clock_map_from_tones(tone["start_time"], onsets[0], tone["end_time"], onsets[-1])


def main(args: Args) -> None:
    ffmpeg = _require_tool("ffmpeg")
    ffprobe = _require_tool("ffprobe")
    record = _pause_timing.load_timing(args.timing_json)
    pauses = _pause_timing.pauses_from_timing(record, args.pause_threshold_s)
    if not pauses:
        raise ValueError(f"{args.timing_json}: no pauses above the threshold; nothing to cut.")

    fps, duration_s, has_audio = probe_video(ffprobe, args.camera_video)
    print(f"Camera video: {fps:.3f} fps, {duration_s:.2f} s, audio={'yes' if has_audio else 'no'}")
    clock_map = build_clock_map(args, record, ffmpeg, has_audio)

    energy: np.ndarray | None = None
    if args.refine_offset or args.motion_check:
        print("Decoding camera video for motion energy...")
        energy = read_motion_energy(ffmpeg, args.camera_video, fps)
    if args.refine_offset:
        assert energy is not None
        clock_map, delta = _pause_timing.refine_clock_map(
            clock_map, pauses, energy, fps, search_window_s = args.refine_window_s, step_s = 0.25 / fps
        )
        print(f"Refined clock offset by {delta * 1000:+.1f} ms against camera motion energy")

    first_step_camera = float(clock_map.to_camera(record["steps"]["sent"][0]))
    last_step_camera = float(clock_map.to_camera(record["steps"]["returned"][-1]))
    if first_step_camera < 0 or last_step_camera > duration_s:
        raise ValueError(
            f"Episode maps to camera time [{first_step_camera:.2f}, {last_step_camera:.2f}] s but the video is "
            f"{duration_s:.2f} s long; the sync is wrong or the recording does not cover the episode."
        )

    intervals = _pause_timing.cut_intervals(pauses, clock_map, fps, args.margin_frames, duration_s)
    output = args.output or str(pathlib.Path(args.camera_video).with_name(pathlib.Path(args.camera_video).stem + "_cut.mp4"))

    flagged = 0
    per_interval_energy: list[float] = []
    if args.motion_check:
        assert energy is not None
        per_interval_energy = [
            _pause_timing.mean_energy_in_intervals(energy, fps, [(start, end)]) for start, end, _ in intervals
        ]
        # A still robot gives every pause roughly the sensor-noise floor, so a pause well above
        # the median of its peers is the one with something moving in the shot.
        pause_median = float(np.median(per_interval_energy))
        for (start, end, pause), value in zip(intervals, per_interval_energy, strict = True):
            if value > 3.0 * pause_median:
                flagged += 1
                print(f"  WARNING: motion inside pause at step {pause.step} ({start:.2f}-{end:.2f} s): energy {value:.2f} vs pause median {pause_median:.2f}")

    cuts_csv = output + ".cuts.csv"
    with open(cuts_csv, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "is_replan", "wall_start", "wall_end", "camera_start", "camera_end", "cut_s", "motion_energy"])
        for i, (start, end, pause) in enumerate(intervals):
            motion = per_interval_energy[i] if per_interval_energy else ""
            writer.writerow([pause.step, int(pause.is_replan), f"{pause.start:.4f}", f"{pause.end:.4f}", f"{start:.4f}", f"{end:.4f}", f"{end - start:.4f}", motion])

    total_cut = sum(end - start for start, end, _ in intervals)
    print(f"{len(intervals)} cuts totalling {total_cut:.1f} s of {last_step_camera - first_step_camera:.1f} s episode ({flagged} flagged)")
    print(f"Cut list: {cuts_csv}")

    with_audio = args.keep_audio and has_audio
    filter_script = output + ".filter"
    write_filter_script(filter_script, [(s, e) for s, e, _ in intervals], with_audio = with_audio)
    command = [ffmpeg, "-y", "-v", "warning", "-stats", "-i", args.camera_video, "-filter_complex_script", filter_script, "-map", "[v]"]
    if with_audio:
        command += ["-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    command += ["-c:v", "libx264", "-crf", str(args.crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", output]
    print("ffmpeg command:\n  " + " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check = True)
    os.remove(filter_script)
    print(f"Wrote {output}")


if __name__ == "__main__":
    tyro.cli(main)
