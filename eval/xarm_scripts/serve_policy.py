"""HTTP server for remote π₀ policy inference.

Loads a trained policy checkpoint and exposes it over a REST API so that a
separate robot machine can request action predictions without needing GPU access.

The server accepts observations (robot state + camera images + subtask prompt)
as JSON and returns the full action chunk produced by the policy.

Usage:
    uv run eval/xarm_scripts/serve_policy.py \
        --config-name cosmos_robocoin_bc_flow \
        --checkpoint-dir /path/to/checkpoint
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import threading
import time
from typing import Any

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from flask import Flask, jsonify, request
from flask_cors import CORS

import openpi.models.model as _model
import openpi.training.config as _config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================


@dataclasses.dataclass
class Args:
    config_name: str
    """TrainConfig name registered in config.py (e.g. 'cosmos_robocoin_bc_flow')."""

    checkpoint_dir: str
    """Path to the policy checkpoint directory."""

    port: int = 8080
    """HTTP port to listen on."""

    host: str = "0.0.0.0"
    """Interface to bind to."""


# =============================================================================
# Policy server
# =============================================================================


class PolicyServer:
    """Loads a π₀ policy checkpoint and serves single-observation inference requests."""

    def __init__(self, config_name: str, checkpoint_dir: str) -> None:
        self._rng_lock = threading.Lock()
        self._load_policy(config_name, checkpoint_dir)

    def _load_policy(self, config_name: str, checkpoint_dir: str) -> None:
        #Code here copied from line 257-338 of computer_counterfactual_actions.py
        from etils import epath
        from flax import nnx
        import orbax.checkpoint as ocp

        import openpi.policies.policy as _policy_module
        import openpi.shared.nnx_utils as _nnx_utils
        import openpi.transforms as _transforms
        from openpi.training import checkpoints as _checkpoints

        config = _config.get_config(config_name)
        policy_model_config = config.policy if config.policy is not None else config.model

        logger.info(f"Loading policy from {checkpoint_dir}")
        checkpoint_dir_path = epath.Path(checkpoint_dir)
        all_params = _model.restore_params(checkpoint_dir_path / "params", dtype=jnp.bfloat16)

        # For actor-critic checkpoints, params are structured as {"policy": ..., "critic": ...}.
        # Extract the policy subtree so we can load it into the BC policy model config.
        if "policy" in all_params:
            policy_params = all_params["policy"]
            if "params" in policy_params and len(policy_params) == 1:
                policy_params = policy_params["params"]
            logger.info(f"Extracted policy params with keys: {list(policy_params.keys())[:10]}")
        else:
            policy_params = all_params

        model_instance = nnx.eval_shape(policy_model_config.create, jax.random.key(0))
        graphdef, state = nnx.split(model_instance)
        policy_params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), policy_params)
        _nnx_utils.replace_state_from_pure_dict_numeric_key_compat(state, policy_params)
        model = nnx.merge(graphdef, state)

        data_config = config.data.create(config.assets_dirs, policy_model_config)
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir_path / "assets", data_config.asset_id)

        policy = _policy_module.Policy(
            model,
            transforms=[
                _transforms.InjectDefaultPrompt(None),
                *data_config.data_transforms.inputs,
                _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *(
                    [_transforms.Clip(data_config.clip_normalized_bounds)]
                    if data_config.clip_normalized_bounds is not None
                    else []
                ),
                *data_config.model_transforms.inputs,
            ],
            output_transforms=[
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ],
            metadata=config.policy_metadata,
        )

        self._model = policy._model  # noqa: SLF001
        self._input_transform = policy._input_transform  # noqa: SLF001
        self._output_transform = policy._output_transform  # noqa: SLF001
        self._sample_kwargs = dict(policy._sample_kwargs)  # noqa: SLF001
        self._rng = policy._rng  # noqa: SLF001

        if hasattr(self._model, "config") and hasattr(self._model.config, "guidance"):
            object.__setattr__(self._model.config, "guidance", 0.0)
            logger.info("Disabled classifier-free guidance (guidance=0.0).")

        from openpi.shared import nnx_utils
        self._sample_actions_jit = nnx_utils.module_jit(self._model.sample_actions)
        logger.info("Policy loaded and JIT-compiled successfully.")


    def predict(self, obs_dict: dict[str, Any]) -> np.ndarray:
        """Run policy inference on a single observation.

        Args:
            obs_dict: Raw observation with keys: image (dict of camera arrays),
                state, prompt, embodiment.

        Returns:
            actions: float32 array of shape [action_horizon, action_dim].
        """
        raw_state = np.asarray(obs_dict["state"], dtype=np.float32)
        transformed = self._input_transform(obs_dict)

        batched = {
            k: jnp.asarray(v)[None, ...]
            for k, v in transformed.items()
            if not isinstance(v, str)
        }

        observation = _model.Observation.from_dict(batched)
        transition = _model.wrap_observation_as_transition(observation)

        with self._rng_lock:
            self._rng, sample_rng = jax.random.split(self._rng)

        actions_out = self._sample_actions_jit(sample_rng, transition, **self._sample_kwargs)
        actions_out = jax.block_until_ready(actions_out)

        actions_np = np.asarray(actions_out[0])  # [action_horizon, action_dim]
        decoded = self._output_transform({
            "embodiment": obs_dict.get("embodiment", ""),
            "state": raw_state,
            "actions": actions_np,
            "next_state": raw_state,
            "next_actions": actions_np,
        })
        return np.asarray(decoded["actions"], dtype=np.float32)


# =============================================================================
# Flask app
# =============================================================================

app = Flask(__name__)
CORS(app)

_server: PolicyServer | None = None


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "No JSON body provided"}), 400

    try:
        state = np.array(data["state"], dtype=np.float32)
        prompt = data["prompt"]
        embodiment = data.get("embodiment", "")

        images = {}
        for cam_name, b64_str in data["images"].items():
            img_bytes = base64.b64decode(b64_str)
            img_np = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            images[cam_name] = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        obs_dict = {
            "image": images,
            "state": state,
            "prompt": prompt,
            "embodiment": embodiment,
        }

        t0 = time.perf_counter()
        actions = _server.predict(obs_dict)
        elapsed = time.perf_counter() - t0

        logger.info(f"Inference completed in {elapsed:.3f}s")
        return jsonify({"actions": actions.tolist(), "inference_time": elapsed})

    except Exception as e:
        logger.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# Entrypoint
# =============================================================================


def main(args: Args) -> None:
    global _server
    logger.info(f"Loading policy: config={args.config_name}, checkpoint={args.checkpoint_dir}")
    _server = PolicyServer(args.config_name, args.checkpoint_dir)
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    tyro.cli(main)
