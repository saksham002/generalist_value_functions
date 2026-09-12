"""Wall-clock timing of the eval control loop, for cutting inference pauses out of external video.

The websocket eval clients block on the policy server every ``query_freq`` steps. While they
block, no command reaches the arm, so the robot holds still for the whole round trip. A camera
on a tripod records those holds; the robot server's own video does not, because it is
step-indexed. This module records when every step was sent and returned so that
``eval/xarm_scripts/tpu_eval/cut_pauses.py`` can locate the holds on the camera's timeline and
cut them.

The camera's clock is not synchronised with the client's. ``PauseTimingLogger`` therefore plays
a short sine tone through the client machine's speaker at the first step of the episode and
again after the last, and records the wall-clock time of each. The camera microphone picks the
tones up, which gives the offset (and drift) between the two clocks without any other link. The
pure functions below (pause derivation, tone detection, clock mapping) are shared with the cut
script and covered by ``pause_timing_test.py``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import subprocess
import time
import wave

import numpy as np

logger = logging.getLogger(__name__)

TIMING_FORMAT_VERSION = 1
TIMING_FILENAME_TEMPLATE = "episode_{episode_idx}_pause_timing.json"
SYNC_TONE_FILENAME = "sync_tone.wav"


@dataclasses.dataclass(frozen = True)
class SyncTone:
    """Sine tone played at episode start and end so the camera audio can be aligned to the log."""

    frequency_hz: float = 1000.0
    duration_s: float = 0.25
    sample_rate_hz: int = 44100
    amplitude: float = 0.6

    def samples(self) -> np.ndarray:
        """Return the tone as int16 PCM with a 5 ms raised-cosine fade at each end."""
        num_samples = int(round(self.duration_s * self.sample_rate_hz))
        t = np.arange(num_samples, dtype = np.float64) / self.sample_rate_hz
        signal = np.sin(2.0 * np.pi * self.frequency_hz * t)
        fade_samples = min(int(0.005 * self.sample_rate_hz), num_samples // 2)
        if fade_samples > 0:
            fade = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_samples)))
            signal[:fade_samples] *= fade
            signal[num_samples - fade_samples:] *= fade[::-1]
        return (self.amplitude * 32767.0 * signal).astype(np.int16)

    def write_wav(self, path: str) -> None:
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate_hz)
            wav_file.writeframes(self.samples().tobytes())


@dataclasses.dataclass(frozen = True)
class Pause:
    """A gap in the control loop, in client wall-clock seconds (``time.time()``)."""

    step: int
    """Env step whose send was delayed; the pause sits between step - 1 returning and this step being sent."""

    start: float
    end: float
    is_replan: bool

    @property
    def duration(self) -> float:
        return self.end - self.start


def derive_pauses(
    steps: np.ndarray,
    sent: np.ndarray,
    returned: np.ndarray,
    replan_steps: set[int],
    pause_threshold_s: float,
) -> list[Pause]:
    """Find every gap between one step returning and the next being sent that exceeds the threshold.

    Deriving pauses from the full step timeline rather than from the inference timer alone means
    that stalls of any origin (image preprocessing, HTTP retries, a manual Enter press) are cut
    too, and that the threshold can be re-chosen offline.
    """
    if not (steps.shape == sent.shape == returned.shape):
        raise ValueError(f"steps, sent and returned must have equal shapes; got {steps.shape}, {sent.shape}, {returned.shape}")
    pauses: list[Pause] = []
    for i in range(1, len(steps)):
        gap = float(sent[i] - returned[i - 1])
        if gap > pause_threshold_s:
            pauses.append(Pause(step = int(steps[i]), start = float(returned[i - 1]), end = float(sent[i]), is_replan = int(steps[i]) in replan_steps))
    return pauses


class PauseTimingLogger:
    """Records wall-clock send/return times of every env step and writes one JSON per episode.

    Call ``mark_step_sent`` immediately before ``env.step`` and ``mark_step_returned`` immediately
    after it returns; call ``mark_replan`` after each policy query. The start tone is played on the
    first ``mark_step_sent`` of the episode and the end tone in ``finish_episode``.
    """

    def __init__(
        self,
        output_dir: str,
        control_freq: int,
        query_freq: int,
        *,
        play_tones: bool = True,
        tone: SyncTone = SyncTone(),
        pause_threshold_periods: float = 2.0,
    ) -> None:
        self.output_dir = output_dir
        self.control_freq = control_freq
        self.query_freq = query_freq
        self.tone = tone
        self.pause_threshold_s = pause_threshold_periods / control_freq
        os.makedirs(output_dir, exist_ok = True)

        self._tone_path = os.path.join(output_dir, SYNC_TONE_FILENAME)
        self._player = shutil.which("aplay") if play_tones else None
        if play_tones and self._player is None:
            logger.warning(
                "PauseTimingLogger: `aplay` not found, sync tones disabled. Camera alignment will need "
                "--sync manual with the camera time of the first robot motion."
            )
        if self._player is not None:
            tone.write_wav(self._tone_path)
        self._reset()

    def _reset(self) -> None:
        self._episode_idx: int | None = None
        self._episode_start: float | None = None
        self._start_tone_time: float | None = None
        self._end_tone_time: float | None = None
        self._steps: list[int] = []
        self._sent: list[float] = []
        self._returned: list[float] = []
        self._replans: list[dict[str, float | int | None]] = []

    def start_episode(self, episode_idx: int) -> None:
        self._reset()
        self._episode_idx = episode_idx
        self._episode_start = time.time()

    def _play_tone(self, blocking: bool) -> float:
        """Start tone playback and return the wall-clock time just before the player was launched.

        The player's own start-up latency is a systematic offset of a few tens of milliseconds that
        the cut script removes by refining the offset against the camera's motion energy.
        """
        assert self._player is not None
        started_at = time.time()
        command = [self._player, "-q", self._tone_path]
        if blocking:
            subprocess.run(command, check = False, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
        else:
            subprocess.Popen(command, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
        return started_at

    def mark_step_sent(self, step: int) -> None:
        if self._episode_idx is None:
            raise RuntimeError("start_episode() must be called before mark_step_sent().")
        if not self._steps and self._player is not None:
            self._start_tone_time = self._play_tone(blocking = False)
        self._steps.append(step)
        self._sent.append(time.time())

    def mark_step_returned(self, step: int) -> None:
        if not self._steps or self._steps[-1] != step:
            raise RuntimeError(f"mark_step_returned({step}) without a matching mark_step_sent().")
        if len(self._returned) != len(self._sent) - 1:
            raise RuntimeError(f"mark_step_returned({step}) called twice for the same step.")
        self._returned.append(time.time())

    def mark_replan(self, step: int, client_seconds: float, server_ms: float | None) -> None:
        self._replans.append({"step": step, "client_seconds": client_seconds, "server_ms": server_ms})

    def finish_episode(self) -> str | None:
        """Play the end tone, write the JSON, and return its path (None if no step was recorded)."""
        if self._episode_idx is None or not self._steps:
            return None
        if self._player is not None:
            self._end_tone_time = self._play_tone(blocking = True)
        if len(self._returned) == len(self._sent) - 1:
            # The last step never returned (exception or Ctrl-C mid-request); drop it so the
            # send/return arrays stay aligned.
            self._steps.pop()
            self._sent.pop()

        steps = np.asarray(self._steps, dtype = np.int64)
        sent = np.asarray(self._sent, dtype = np.float64)
        returned = np.asarray(self._returned, dtype = np.float64)
        replan_steps = {int(r["step"]) for r in self._replans}
        pauses = derive_pauses(steps, sent, returned, replan_steps, self.pause_threshold_s)

        record = {
            "format_version": TIMING_FORMAT_VERSION,
            "episode_idx": self._episode_idx,
            "control_freq": self.control_freq,
            "query_freq": self.query_freq,
            "pause_threshold_s": self.pause_threshold_s,
            "episode_start": self._episode_start,
            "tone": {
                "frequency_hz": self.tone.frequency_hz,
                "duration_s": self.tone.duration_s,
                "start_time": self._start_tone_time,
                "end_time": self._end_tone_time,
            },
            "replans": self._replans,
            "steps": {"step": steps.tolist(), "sent": sent.tolist(), "returned": returned.tolist()},
            "pauses": [dataclasses.asdict(p) for p in pauses],
        }
        path = os.path.join(self.output_dir, TIMING_FILENAME_TEMPLATE.format(episode_idx = self._episode_idx))
        with open(path, "w") as f:
            json.dump(record, f)
        total_pause = sum(p.duration for p in pauses)
        wall = float(returned[-1] - sent[0]) if len(sent) > 0 else 0.0
        logger.info(
            f"Saved pause timing: {path} ({len(pauses)} pauses, {total_pause:.1f}s of {wall:.1f}s wall time"
            f"{'' if self._player is not None else ', no sync tones'})"
        )
        return path


# =============================================================================
# Offline: tone detection and clock mapping (used by cut_pauses.py)
# =============================================================================


def load_timing(path: str) -> dict:
    with open(path) as f:
        record = json.load(f)
    version = record.get("format_version")
    if version != TIMING_FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported pause-timing format version {version!r} (expected {TIMING_FORMAT_VERSION}).")
    return record


def pauses_from_timing(record: dict, pause_threshold_s: float | None = None) -> list[Pause]:
    """Re-derive the pause list from the step timeline, optionally with a different threshold."""
    steps = np.asarray(record["steps"]["step"], dtype = np.int64)
    sent = np.asarray(record["steps"]["sent"], dtype = np.float64)
    returned = np.asarray(record["steps"]["returned"], dtype = np.float64)
    replan_steps = {int(r["step"]) for r in record["replans"]}
    threshold = record["pause_threshold_s"] if pause_threshold_s is None else pause_threshold_s
    return derive_pauses(steps, sent, returned, replan_steps, threshold)


def tone_band_purity(
    audio: np.ndarray,
    sample_rate_hz: int,
    frequency_hz: float,
    window_s: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample fraction of energy at ``frequency_hz`` within a sliding window, plus the band power.

    A pure tone scores close to 1 regardless of loudness, and speech, motor noise and clicks
    score low, which is what makes the detector robust to the recording level.
    """
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1-D mono; got shape {audio.shape}")
    audio = audio.astype(np.float64)
    window = max(int(round(window_s * sample_rate_hz)), 1)
    t = np.arange(audio.shape[0], dtype = np.float64) / sample_rate_hz
    demodulated = audio * np.exp(-2j * np.pi * frequency_hz * t)

    def _moving_sum(values: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(np.concatenate([np.zeros(1, dtype = values.dtype), values]))
        summed = cumulative[window:] - cumulative[:-window]
        # Right-pad so the output is aligned to the window's first sample.
        return np.concatenate([summed, np.full(window - 1, summed[-1] if summed.shape[0] else 0, dtype = values.dtype)])

    band_power = np.abs(_moving_sum(demodulated)) ** 2 / window
    total_power = _moving_sum(audio ** 2) + 1e-12
    # The demodulated moving sum estimates the tone's amplitude a via |sum| ~= a*window/2, so its
    # power is a²/4 * window; the tone's mean-square is a²/2, hence the factor 2.
    purity = np.clip(2.0 * band_power / total_power, 0.0, 1.0)
    return purity, total_power / window


def detect_tones(
    audio: np.ndarray,
    sample_rate_hz: int,
    frequency_hz: float,
    duration_s: float,
    *,
    purity_threshold: float = 0.6,
    min_level_ratio: float = 4.0,
) -> list[float]:
    """Return the onset time (s) of every tone burst found in ``audio``.

    An onset is the first sample of a run where the band purity stays above the threshold for at
    least half the tone's duration and the level is well above the recording's median level.
    """
    purity, level = tone_band_purity(audio, sample_rate_hz, frequency_hz)
    noise_floor = float(np.median(level)) + 1e-12
    active = (purity > purity_threshold) & (level > min_level_ratio * noise_floor)
    min_run = int(0.5 * duration_s * sample_rate_hz)
    padded = np.concatenate([[False], active, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts = edges[0::2]
    ends = edges[1::2]
    onsets = [float(s) / sample_rate_hz for s, e in zip(starts, ends, strict = True) if e - s >= min_run]
    return onsets


@dataclasses.dataclass(frozen = True)
class ClockMap:
    """Affine map from client wall-clock seconds to camera-timeline seconds."""

    wall_reference: float
    camera_reference: float
    scale: float = 1.0

    def to_camera(self, wall_time: float | np.ndarray) -> float | np.ndarray:
        return self.camera_reference + (wall_time - self.wall_reference) * self.scale

    def shifted(self, camera_delta_s: float) -> ClockMap:
        return dataclasses.replace(self, camera_reference = self.camera_reference + camera_delta_s)


def clock_map_from_tones(
    start_tone_wall: float,
    start_tone_camera: float,
    end_tone_wall: float | None,
    end_tone_camera: float | None,
) -> ClockMap:
    """Build the wall-to-camera map from the start tone, using the end tone for drift when present."""
    scale = 1.0
    if end_tone_wall is not None and end_tone_camera is not None:
        wall_span = end_tone_wall - start_tone_wall
        if wall_span <= 0:
            raise ValueError(f"end tone ({end_tone_wall}) is not after the start tone ({start_tone_wall}).")
        scale = (end_tone_camera - start_tone_camera) / wall_span
        if not 0.99 < scale < 1.01:
            raise ValueError(
                f"Camera/client clock ratio {scale:.5f} is implausible; the detected tones probably "
                "do not match the logged ones."
            )
    return ClockMap(wall_reference = start_tone_wall, camera_reference = start_tone_camera, scale = scale)


def cut_intervals(
    pauses: list[Pause],
    clock_map: ClockMap,
    fps: float,
    margin_frames: float,
    video_duration_s: float,
) -> list[tuple[float, float, Pause]]:
    """Map each pause into camera time, shrink it by the margin, and drop any that is too short to cut."""
    margin_s = margin_frames / fps
    intervals: list[tuple[float, float, Pause]] = []
    for pause in pauses:
        start = float(clock_map.to_camera(pause.start)) + margin_s
        end = float(clock_map.to_camera(pause.end)) - margin_s
        start = max(start, 0.0)
        end = min(end, video_duration_s)
        if end - start >= 1.0 / fps:
            intervals.append((start, end, pause))
    return intervals


def frame_motion_energy(frames: np.ndarray) -> np.ndarray:
    """Mean absolute difference between consecutive grayscale frames; index i is the motion into frame i."""
    if frames.ndim != 3:
        raise ValueError(f"frames must be (num_frames, height, width); got shape {frames.shape}")
    diffs = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
    energy = diffs.reshape(diffs.shape[0], diffs.shape[1] * diffs.shape[2]).mean(axis = 1)
    return np.concatenate([np.zeros(1, dtype = energy.dtype), energy])


def mean_energy_in_intervals(energy: np.ndarray, fps: float, intervals: list[tuple[float, float]]) -> float:
    """Average motion energy over all frames whose timestamps fall inside any of the intervals."""
    selected: list[np.ndarray] = []
    for start, end in intervals:
        first = int(np.ceil(start * fps))
        last = int(np.floor(end * fps))
        if last >= first:
            selected.append(energy[max(first, 0): min(last + 1, energy.shape[0])])
    if not selected:
        return float("nan")
    return float(np.concatenate(selected).mean())


def refine_clock_map(
    clock_map: ClockMap,
    pauses: list[Pause],
    energy: np.ndarray,
    fps: float,
    search_window_s: float,
    step_s: float,
) -> tuple[ClockMap, float]:
    """Shift the map within +-search_window_s to minimise motion energy inside the pauses.

    This absorbs the tone player's start-up latency and any residual sync error: the correct
    offset is the one at which the predicted holds line up with the camera's still frames.
    Returns the refined map and the shift applied.
    """
    if not pauses:
        return clock_map, 0.0
    best_delta = 0.0
    best_score = float("inf")
    duration_s = energy.shape[0] / fps
    for delta in np.arange(-search_window_s, search_window_s + 0.5 * step_s, step_s):
        candidate = clock_map.shifted(float(delta))
        intervals = [(s, e) for s, e, _ in cut_intervals(pauses, candidate, fps, margin_frames = 0.0, video_duration_s = duration_s)]
        score = mean_energy_in_intervals(energy, fps, intervals)
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_delta = float(delta)
    return clock_map.shifted(best_delta), best_delta
