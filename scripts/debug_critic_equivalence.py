"""Critic code-equivalence harness (OLD vs NEW openpi code).

Loads the SERVED critic checkpoint under whatever openpi code is on PYTHONPATH,
feeds a FIXED, seeded input, and prints the critic Q-value(s) plus
``observation.subtask_start_index``.

Purpose
-------
Confirm that the OLD (pre-feature-pull) and NEW (feature) openpi code produce
the *same* critic output for the *same* input and *same* checkpoint. Since the
checkpoint is identical across runs, the weights are identical, so any
difference in the printed Q-values is caused purely by the code diff.

It also prints ``subtask_start_index``: at serve time it is expected to be
``None`` (the policy uses ``TokenizePrompt``), which is the single assumption
the whole static analysis rests on. This run confirms it on the real model.

How to run (on the cluster, with two code checkouts available)
--------------------------------------------------------------
    OLD=/path/to/batch_value_learning          # the `old-code-baseline` checkout
    NEW=/path/to/batch_value_learning_new      # an `origin/main` (feature) checkout

    PYTHONPATH=$OLD/src python scripts/debug_critic_equivalence.py | tee /tmp/q_old.txt
    PYTHONPATH=$NEW/src python scripts/debug_critic_equivalence.py | tee /tmp/q_new.txt
    diff /tmp/q_old.txt /tmp/q_new.txt

Interpretation
--------------
* ``subtask=None(serve)`` rows identical between OLD and NEW  -> serve path is
  code-equivalent (expected).
* ``subtask=set(...)`` rows differ                            -> confirms the
  +2/+1 RoPE-shift change is real but only active when subtask indices are set.
* ``subtask_start_index = None`` printed                      -> confirms the
  serve assumption end-to-end.

Notes
-----
* Single-device node: pass ``--fsdp-devices 1`` (default). The checkpoint was
  trained with fsdp=16; loading reshards onto the available mesh.
* This is a debugging tool; if a shape/arg mismatch appears on your cluster
  build, the diagnostic prints (shapes, device count, module path) make it easy
  to adjust. It does NOT need to "look right" numerically — only to be IDENTICAL
  across the two code versions for the same input.
"""

from __future__ import annotations

import argparse

import numpy as np
import jax

import openpi.models.model as _model
from openpi.robocoin_utils.load_model_utils import load_critic

# Mirrors eval/xarm_scripts/tpu_eval/serve_policy_shirt_hang.sh
CRITIC_CONFIG = "robocoin_bimanual_paligemma_cql_rlds_subtask_no_ntp"
CRITIC_DIR = (
    "gs://saksham-euw4/checkpoints/robocoin/value_functions/Q/"
    "robocoin_bimanual_paligemma_cql_rlds_subtask_no_ntp/"
    "robocoin_bimanual_paligemma_cql_rlds_subtask_no_ntp/"
    "real_shirt_hang_paligemma_cql_rlds_finetune_subtask_final"
)
CRITIC_FT = "real_shirt_hang_paligemma_cql_rlds_finetune_subtask_final"
CRITIC_STEP = 250000


def _net_config(cfg):
    """The PaliGemma network config lives under q_network_config (CQL) or network_config (SARSA)."""
    return getattr(cfg.model, "q_network_config", None) or cfg.model.network_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsdp-devices", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--critic-step", type=int, default=CRITIC_STEP)
    args = ap.parse_args()

    print(f"[harness] openpi module path: {_model.__file__}")
    print(f"[harness] jax devices: {jax.device_count()} ({jax.devices()})")

    critic_model, norm_stats, cfg, step = load_critic(
        CRITIC_CONFIG,
        CRITIC_DIR,
        fine_tune=CRITIC_FT,
        step=args.critic_step,
        fsdp_devices=args.fsdp_devices,
    )
    print(f"[harness] loaded critic '{CRITIC_CONFIG}' (FT={CRITIC_FT}) step={step}")

    net = _net_config(cfg)
    num_cameras = net.num_cameras
    img = net.image_size
    height = img[0] if isinstance(img, (list, tuple)) else img
    width = img[1] if isinstance(img, (list, tuple)) else img
    max_token_len = net.max_token_len
    state_dim = net.state_dim
    action_dim = net.action_dim
    action_horizon = cfg.action_horizon or cfg.model.action_horizon
    batch = 1
    print(
        f"[harness] shapes: cams={num_cameras} HxW={height}x{width} tok={max_token_len} "
        f"state={state_dim} act_dim={action_dim} act_horizon={action_horizon}"
    )

    rng = np.random.default_rng(args.seed)
    cam_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"][:num_cameras]
    image = {k: rng.integers(0, 256, size=(batch, height, width, 3)).astype(np.uint8) for k in cam_keys}
    image_mask = {k: np.ones((batch,), dtype=bool) for k in cam_keys}
    state = rng.standard_normal((batch, state_dim)).astype(np.float32)
    tokens = rng.integers(0, 1000, size=(batch, max_token_len)).astype(np.int32)
    tok_mask = np.ones((batch, max_token_len), dtype=bool)
    action_mask = np.ones((batch, action_horizon), dtype=bool)
    # Actions kept inside the critic's normalized action bounds.
    action = rng.uniform(-1.0, 1.0, size=(batch, action_horizon, action_dim)).astype(np.float32)

    def build_obs(subtask):
        data = {
            "image": {k: v.copy() for k, v in image.items()},
            "image_mask": dict(image_mask),
            "state": state.copy(),
            "tokenized_prompt": tokens.copy(),
            "tokenized_prompt_mask": tok_mask.copy(),
            "action_mask": action_mask.copy(),
        }
        if subtask is not None:
            data["subtask_start_index"] = np.asarray([subtask[0]] * batch, dtype=np.int32)
            data["subtask_end_index"] = np.asarray([subtask[1]] * batch, dtype=np.int32)
        return _model.Observation.from_dict(data)

    for label, subtask in [("subtask=None(serve)", None), ("subtask=set(5,8)", (5, 8))]:
        obs = build_obs(subtask)
        print(f"\n=== {label} ===")
        print(f"  observation.subtask_start_index = {np.asarray(obs.subtask_start_index).tolist() if obs.subtask_start_index is not None else None}")

        q_full = critic_model.compute_value(obs, action)
        q_full = q_full[0] if isinstance(q_full, tuple) else q_full
        print(f"  Q[full-forward] = {np.asarray(q_full).ravel().tolist()}")

        # KV-cache path mirrors the real BestOfN serve forward (compute_prefix_cache
        # + cached suffix). use_target=True matches use_target_value=True in the config.
        try:
            cache = critic_model.compute_prefix_cache(obs, use_target=True)
            q_kv = critic_model.compute_value(obs, action, prefix_cache=cache)
            q_kv = q_kv[0] if isinstance(q_kv, tuple) else q_kv
            print(f"  Q[kv-cache]     = {np.asarray(q_kv).ravel().tolist()}")
        except Exception as e:
            print(f"  Q[kv-cache]     = (skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
