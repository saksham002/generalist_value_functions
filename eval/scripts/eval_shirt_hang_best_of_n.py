#!/usr/bin/env python3
"""Best-of-N real-world evaluation for the shirt-hang task."""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
import sys
from typing import Annotated

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import tyro

from openpi.models import model as _model
from openpi.training import config as _value_config
from openpi.training import config_shirt_hang as _policy_config_mod

from eval.extra_models.best_of_n_dual_obs import DualObservationBestOfNWrapper
from eval.helpers import candidate_actions as _candidate_actions
from eval.helpers import checkpoint_loading as _checkpoint_loading
from eval.helpers import preprocessing as _preprocessing
from eval.helpers import runtime_types
from eval.helpers import utils as _utils
from eval.helpers.subtask_tracking import ShirtHangSubtaskDetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
possible_rac_roots: list[Path] = []
if env_path := os.environ.get("RAC_PATH"):
    possible_rac_roots.append(Path(env_path))
possible_rac_roots.append(PROJECT_ROOT.parent / "RAC")
possible_rac_roots.append(Path.cwd().parent / "RAC")

RAC_ROOT: Path | None = None
for candidate_root in possible_rac_roots:
    if candidate_root.exists():
        RAC_ROOT = candidate_root
        for candidate in (candidate_root, candidate_root / "dual_xarms"):
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))
        break

try:
    from dual_xarms_sim.relative_frame import RelativeFrame
    from dual_xarms_sim.relative_frame import WristRelativeTo
    from dual_xarms_sim.rmp_env import RMPDualXArmsEnv

    try:
        from flow_match.bc_flowmatch.env_wrappers.frame_stack_wrapper import FrameStackWrapperEnv
    except ImportError:
        FrameStackWrapperEnv = None
except ImportError as exc:
    raise ImportError("Could not import RMPDualXArmsEnv. Check RAC_PATH.") from exc


class ShirtHangObservationFactory:
    """Build policy-facing and critic-facing raw inputs from env observations."""

    camera_map = {
        "right/top": "base_0_rgb",
        "left/wrist": "left_wrist_0_rgb",
        "right/wrist": "right_wrist_0_rgb",
    }

    def __init__(self, camera_names: tuple[str, ...]):
        self._camera_names = camera_names

    def build(self, env_obs: dict, prompt: str) -> runtime_types.EvalObservationPair:
        images = self._extract_images(env_obs)
        image_mask = {name: np.array(True, dtype = bool) for name in images}
        return runtime_types.EvalObservationPair(
            policy_inputs = {
                "image": images,
                "image_mask": image_mask,
                "state": self._extract_policy_state(env_obs),
            },
            critic_inputs = {
                "image": images,
                "image_mask": image_mask,
                "state": self._extract_critic_state(env_obs),
                "prompt": prompt,
                "action_mask": np.ones((30,), dtype = bool),
            },
            prompt = prompt,
        )

    def _extract_images(self, env_obs: dict) -> dict[str, np.ndarray]:
        images = {}
        for camera_name in self._camera_names:
            canonical_name = self.camera_map[camera_name]
            image_source = env_obs["images"][canonical_name] if canonical_name in env_obs["images"] else env_obs["images"][camera_name]
            images[canonical_name] = _utils.latest_rgb_frame(image_source)
        return images

    def _extract_policy_state(self, env_obs: dict) -> np.ndarray:
        state = env_obs["state"]
        parts = [
            _utils.last_array(state["left/relative2_tcp_pose"]).astype(np.float32),
            _utils.last_array(state["left/relative2_tcp_vel"]).astype(np.float32),
            _utils.last_array(state["left/wrist_tcp_vel"]).astype(np.float32),
            _utils.last_array(state["left/gripper_pos"]).astype(np.float32),
            _utils.last_array(state["right/relative2_tcp_pose"]).astype(np.float32),
            _utils.last_array(state["right/relative2_tcp_vel"]).astype(np.float32),
            _utils.last_array(state["right/wrist_tcp_vel"]).astype(np.float32),
            _utils.last_array(state["right/gripper_pos"]).astype(np.float32),
        ]
        return np.concatenate(parts, axis = -1).astype(np.float32)

    def _extract_critic_state(self, env_obs: dict) -> np.ndarray:
        state = env_obs["state"]
        left_tcp_pose = _utils.last_array(state["left/tcp_pose"]).astype(np.float32)
        right_tcp_pose = _utils.last_array(state["right/tcp_pose"]).astype(np.float32)
        left_gripper = _utils.last_array(state["left/gripper_pos"]).astype(np.float32)
        right_gripper = _utils.last_array(state["right/gripper_pos"]).astype(np.float32)
        left_eef = np.concatenate([left_tcp_pose[:3], _utils.quat_to_euler(left_tcp_pose[3:7])], axis = -1)
        right_eef = np.concatenate([right_tcp_pose[:3], _utils.quat_to_euler(right_tcp_pose[3:7])], axis = -1)
        return np.concatenate([left_eef, left_gripper[:1], right_eef, right_gripper[:1]], axis = -1).astype(np.float32)


@dataclasses.dataclass
class EvalArgs:
    policy_config: Annotated[str, tyro.conf.Positional]
    policy_checkpoint: Annotated[str, tyro.conf.Positional]
    value_config: Annotated[str, tyro.conf.Positional]
    value_checkpoint: Annotated[str, tyro.conf.Positional]
    num_samples: int = 8
    num_episodes: int = 10
    query_freq: int = 30
    max_env_steps: int = 7200
    control_hz: int = 60
    time_limit_s: float | None = None
    seed: int = 86
    camera_names: tuple[str, ...] = ("right/top", "left/wrist", "right/wrist")
    remote_robot: bool = False
    robot_host: str = "localhost"
    robot_port: int = 8080
    robot_timeout: float = 30.0
    trace_replans: bool = True


def _load_env(args: EvalArgs, policy_train_config):
    if args.remote_robot:
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from remote_environment_adapter import RemoteEnvironmentAdapter

        return RemoteEnvironmentAdapter(
            host = args.robot_host,
            port = args.robot_port,
            timeout = args.robot_timeout,
        )

    env_kwargs = {
        "control_freq": args.control_hz,
        "time_limit": args.time_limit_s or (args.max_env_steps / args.control_hz),
    }
    env = RMPDualXArmsEnv(**env_kwargs)
    env = RelativeFrame(env)
    env = WristRelativeTo(env)
    if FrameStackWrapperEnv is not None:
        obs_history_len = getattr(policy_train_config.data, "obs_history_len", 1)
        history_gap = getattr(policy_train_config.data, "history_gap", 29)
        env = FrameStackWrapperEnv(env, n_frames = obs_history_len, gap = history_gap)
    return env


def _build_critic_action_transform(
    policy_bundle,
    value_bundle,
    transformed_policy_inputs: dict,
    raw_critic_inputs: dict,
    *,
    real_action_start: int,
    real_action_dim: int,
):
    def _transform(candidate_actions: jnp.ndarray) -> jnp.ndarray:
        candidate_actions_np = np.asarray(candidate_actions)
        batch_size, num_samples, _, _ = candidate_actions_np.shape
        normalized_candidates = np.zeros((batch_size, num_samples, 30, real_action_dim), dtype = np.float32)
        for batch_idx in range(batch_size):
            for sample_idx in range(num_samples):
                decoded = _preprocessing.decode_policy_actions(
                    policy_bundle,
                    transformed_policy_inputs,
                    candidate_actions_np[batch_idx, sample_idx],
                )
                env_actions = _candidate_actions.slice_real_actions(
                    decoded,
                    real_action_start = real_action_start,
                    real_action_dim = real_action_dim,
                )
                critic_actions = _candidate_actions.subsample_even_actions(env_actions, target_length = 30)
                critic_inputs = dict(raw_critic_inputs)
                critic_inputs["actions"] = critic_actions
                _, normalized_action = _preprocessing.prepare_value_inputs(value_bundle, critic_inputs)
                normalized_candidates[batch_idx, sample_idx] = np.asarray(normalized_action[0])
        return jnp.asarray(normalized_candidates)

    return _transform


def _select_best_action_chunk(
    dual_wrapper: DualObservationBestOfNWrapper,
    policy_bundle,
    value_bundle,
    observation_pair: runtime_types.EvalObservationPair,
    rng: jax.Array,
    *,
    real_action_start: int,
    real_action_dim: int,
) -> tuple[np.ndarray, jax.Array, np.ndarray]:
    transformed_policy_inputs, policy_observation = _preprocessing.prepare_policy_observation(
        policy_bundle,
        observation_pair.policy_inputs,
    )
    bootstrap_inputs = dict(observation_pair.critic_inputs)
    bootstrap_inputs["actions"] = np.zeros((30, real_action_dim), dtype = np.float32)
    critic_observation, _ = _preprocessing.prepare_value_inputs(value_bundle, bootstrap_inputs)
    critic_action_transform = _build_critic_action_transform(
        policy_bundle,
        value_bundle,
        transformed_policy_inputs,
        observation_pair.critic_inputs,
        real_action_start = real_action_start,
        real_action_dim = real_action_dim,
    )

    transition = _model.wrap_observation_as_transition(policy_observation)
    rng, sample_rng = jax.random.split(rng)
    selected_actions, candidate_values = dual_wrapper.sample_actions_with_values(
        sample_rng,
        transition,
        value_function = value_bundle.model,
        critic_observation = critic_observation,
        critic_action_transform = critic_action_transform,
        **policy_bundle.sample_kwargs,
    )
    decoded = _preprocessing.decode_policy_actions(
        policy_bundle,
        transformed_policy_inputs,
        np.asarray(selected_actions[0]),
    )
    env_actions = _candidate_actions.slice_real_actions(
        decoded,
        real_action_start = real_action_start,
        real_action_dim = real_action_dim,
    )
    return env_actions, rng, np.asarray(candidate_values[0], dtype = np.float32)


def _debug_policy_enabled() -> bool:
    value = os.environ.get("DEBUG_POLICY", "")
    return value.lower() not in ("", "0", "false", "no")


def evaluate(args: EvalArgs) -> None:
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(args.seed)
    debug_policy = _debug_policy_enabled()

    policy_train_config = _policy_config_mod.get_config(args.policy_config)
    value_train_config = _value_config.get_config(args.value_config)
    policy_bundle = _checkpoint_loading.load_policy_bundle(
        policy_train_config,
        args.policy_checkpoint,
        default_prompt = getattr(policy_train_config.data, "default_prompt", None),
    )
    value_bundle = _checkpoint_loading.load_value_bundle(value_train_config, args.value_checkpoint)

    real_action_start = getattr(policy_train_config.model, "real_action_start", 0)
    real_action_dim = getattr(policy_train_config.model, "real_action_dim", policy_train_config.model.action_dim)
    dual_wrapper = DualObservationBestOfNWrapper(
        action_dim = policy_bundle.model.action_dim,
        action_horizon = policy_bundle.model.action_horizon,
        max_token_len = policy_bundle.model.max_token_len,
        base_model = policy_bundle.model,
        num_samples = args.num_samples,
        take_min_over_ensemble = True,
        use_target_value = False,
        selection_mode = "argmax",
        softmax_temperature = 1.0,
    )
    observation_factory = ShirtHangObservationFactory(args.camera_names)
    detector = ShirtHangSubtaskDetector()
    env = _load_env(args, policy_train_config)
    rng = jax.random.key(args.seed)

    for episode_idx in tqdm(range(args.num_episodes), desc = "Episodes"):
        env_obs, _ = env.reset()
        detector.reset()
        planned_actions: np.ndarray | None = None
        planned_index = 0
        for step_idx in range(args.max_env_steps):
            prompt = detector.update(env_obs, step_idx)
            if planned_actions is None or planned_index >= len(planned_actions) or step_idx % args.query_freq == 0:
                observation_pair = observation_factory.build(env_obs, prompt)
                planned_actions, rng, candidate_values = _select_best_action_chunk(
                    dual_wrapper,
                    policy_bundle,
                    value_bundle,
                    observation_pair,
                    rng,
                    real_action_start = real_action_start,
                    real_action_dim = real_action_dim,
                )
                planned_index = 0
                if args.trace_replans:
                    logging.info(f"[Episode {episode_idx}] step = {step_idx}, prompt = {prompt}")
                if debug_policy and step_idx % 30 == 0:
                    values_str = ", ".join(f"{value:.6f}" for value in candidate_values.tolist())
                    logging.info(f"{prompt}: {values_str}")

            env_action = planned_actions[planned_index]
            env_obs, reward, terminated, truncated, _ = env.step(env_action)
            planned_index += 1
            if terminated or truncated:
                logging.info(
                    f"Episode {episode_idx} finished at step {step_idx} "
                    f"(terminated = {terminated}, truncated = {truncated}, reward = {reward})"
                )
                break

    env.close()


def main() -> None:
    evaluate(tyro.cli(EvalArgs))


if __name__ == "__main__":
    main()
