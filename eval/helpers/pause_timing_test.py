import json
import os

import numpy as np
import pytest

from eval.helpers import pause_timing


def _synthetic_timeline(control_freq: int, query_freq: int, num_steps: int, infer_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    period = 1.0 / control_freq
    steps = np.arange(num_steps, dtype = np.int64)
    sent = np.zeros(num_steps, dtype = np.float64)
    returned = np.zeros(num_steps, dtype = np.float64)
    now = 1_700_000_000.0
    for i in range(num_steps):
        if i % query_freq == 0:
            now += infer_s
        sent[i] = now
        now += period
        returned[i] = now
        now += 0.002
    return steps, sent, returned


def test_derive_pauses_finds_only_replan_gaps():
    steps, sent, returned = _synthetic_timeline(control_freq = 60, query_freq = 30, num_steps = 120, infer_s = 0.8)
    pauses = pause_timing.derive_pauses(steps, sent, returned, replan_steps = {30, 60, 90}, pause_threshold_s = 2.0 / 60)
    assert [p.step for p in pauses] == [30, 60, 90]
    assert all(p.is_replan for p in pauses)
    assert all(abs(p.duration - 0.802) < 1e-6 for p in pauses)


def test_derive_pauses_marks_non_replan_stalls():
    steps, sent, returned = _synthetic_timeline(control_freq = 60, query_freq = 30, num_steps = 60, infer_s = 0.5)
    sent[45:] += 1.0
    returned[45:] += 1.0
    pauses = pause_timing.derive_pauses(steps, sent, returned, replan_steps = {30}, pause_threshold_s = 2.0 / 60)
    assert [(p.step, p.is_replan) for p in pauses] == [(30, True), (45, False)]


def test_detect_tones_finds_two_bursts_in_noise():
    tone = pause_timing.SyncTone(sample_rate_hz = 16000)
    rng = np.random.default_rng(0)
    duration_s = 20.0
    audio = rng.normal(0.0, 300.0, int(duration_s * tone.sample_rate_hz))
    burst = tone.samples().astype(np.float64) * 0.05
    for onset_s in (2.5, 17.25):
        start = int(onset_s * tone.sample_rate_hz)
        audio[start: start + burst.shape[0]] += burst
    onsets = pause_timing.detect_tones(audio, tone.sample_rate_hz, tone.frequency_hz, tone.duration_s)
    assert len(onsets) == 2
    assert abs(onsets[0] - 2.5) < 0.005
    assert abs(onsets[1] - 17.25) < 0.005


def test_detect_tones_ignores_broadband_click():
    sample_rate_hz = 16000
    audio = np.zeros(sample_rate_hz * 5, dtype = np.float64)
    audio[sample_rate_hz * 2: sample_rate_hz * 2 + 400] = 20000.0
    assert pause_timing.detect_tones(audio, sample_rate_hz, 1000.0, 0.25) == []


def test_clock_map_uses_end_tone_for_drift():
    clock_map = pause_timing.clock_map_from_tones(100.0, 5.0, 200.0, 105.1)
    assert clock_map.scale == pytest.approx(1.001)
    assert clock_map.to_camera(150.0) == pytest.approx(55.05)
    assert clock_map.shifted(0.02).to_camera(100.0) == pytest.approx(5.02)


def test_clock_map_rejects_mismatched_tones():
    with pytest.raises(ValueError, match = "implausible"):
        pause_timing.clock_map_from_tones(100.0, 5.0, 200.0, 140.0)


def test_cut_intervals_apply_margin_and_drop_short_pauses():
    pauses = [
        pause_timing.Pause(step = 30, start = 10.0, end = 11.0, is_replan = True),
        pause_timing.Pause(step = 60, start = 20.0, end = 20.05, is_replan = True),
    ]
    clock_map = pause_timing.ClockMap(wall_reference = 0.0, camera_reference = 1.0)
    intervals = pause_timing.cut_intervals(pauses, clock_map, fps = 50.0, margin_frames = 1.0, video_duration_s = 100.0)
    assert len(intervals) == 1
    start, end, pause = intervals[0]
    assert pause.step == 30
    assert start == pytest.approx(11.02)
    assert end == pytest.approx(11.98)


def test_refine_clock_map_recovers_offset_from_motion_energy():
    fps = 50.0
    num_frames = 1000
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 256, size = (num_frames, 8, 8), dtype = np.uint8)
    pauses = []
    for start_s in (4.0, 8.0, 12.0):
        pauses.append(pause_timing.Pause(step = 0, start = start_s, end = start_s + 1.0, is_replan = True))
        first = int((start_s + 0.1) * fps)
        last = int((start_s + 1.1) * fps)
        frames[first: last + 1] = frames[first]
    energy = pause_timing.frame_motion_energy(frames)
    initial = pause_timing.ClockMap(wall_reference = 0.0, camera_reference = 0.0)
    refined, delta = pause_timing.refine_clock_map(initial, pauses, energy, fps, search_window_s = 0.3, step_s = 0.01)
    assert delta == pytest.approx(0.1, abs = 0.011)
    assert refined.to_camera(4.0) == pytest.approx(4.1, abs = 0.011)


def test_logger_writes_json_without_player(tmp_path, monkeypatch):
    monkeypatch.setattr(pause_timing.shutil, "which", lambda name: None)
    logger = pause_timing.PauseTimingLogger(str(tmp_path), control_freq = 60, query_freq = 30, play_tones = True)
    logger.start_episode(3)
    for step in range(4):
        logger.mark_step_sent(step)
        logger.mark_step_returned(step)
    logger.mark_replan(0, 0.5, 400.0)
    path = logger.finish_episode()
    assert path == os.path.join(str(tmp_path), "episode_3_pause_timing.json")
    with open(path) as f:
        record = json.load(f)
    assert record["episode_idx"] == 3
    assert record["tone"]["start_time"] is None
    assert record["steps"]["step"] == [0, 1, 2, 3]
    assert len(record["steps"]["sent"]) == len(record["steps"]["returned"]) == 4
    assert pause_timing.pauses_from_timing(record) == []


def test_logger_drops_unreturned_last_step(tmp_path, monkeypatch):
    monkeypatch.setattr(pause_timing.shutil, "which", lambda name: None)
    logger = pause_timing.PauseTimingLogger(str(tmp_path), control_freq = 60, query_freq = 30)
    logger.start_episode(0)
    logger.mark_step_sent(0)
    logger.mark_step_returned(0)
    logger.mark_step_sent(1)
    record = pause_timing.load_timing(logger.finish_episode())
    assert record["steps"]["step"] == [0]
