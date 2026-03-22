"""Reusable small utilities for eval-time adapters."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def last_array(value: Any) -> np.ndarray:
    """Return the last element for stacked observations, otherwise the value itself."""
    array = np.asarray(value)
    return array[-1] if array.ndim > 1 else array


def quat_to_euler(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion `(x, y, z, w)` to Euler `(roll, pitch, yaw)`."""
    x, y, z, w = np.asarray(quat_xyzw, dtype = np.float32)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype = np.float32)


def bgr_frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert one frame from BGR to RGB and ensure `uint8` output."""
    rgb = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def latest_rgb_frame(frames: np.ndarray) -> np.ndarray:
    """Extract the newest frame from a possibly stacked BGR frame tensor."""
    frame = frames[-1] if np.asarray(frames).ndim == 4 else frames
    return bgr_frame_to_rgb(frame)
