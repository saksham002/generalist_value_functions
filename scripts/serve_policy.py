import dataclasses
import enum
import logging
import os
import socket
import time
from typing import Literal

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    # Optional explicit checkpoint step. Only consulted by the BestOfN
    # critic-aware loading path (when --critic.* is set); the default policy
    # loader points `dir` at a step subdirectory directly.
    step: int | None = None
    # Optional FineTuneConfig name. Only consulted by the BestOfN loading path.
    fine_tune_config: str | None = None


@dataclasses.dataclass
class CriticArgs:
    """Optional critic / value-function args for BestOfN action selection.

    When set, scripts/serve_policy.py loads a critic checkpoint alongside the
    policy and wraps both in BestOfNPolicy. When unset, behavior is byte-
    identical to the policy-only serving path.
    """

    # Critic train config name (e.g., "robocoin_bimanual_paligemma_q_sarsa").
    config: str
    # Critic checkpoint directory.
    dir: str
    # Optional explicit critic step. None → use the latest step subdir.
    step: int | None = None
    # Optional FineTuneConfig name (e.g., "robocasa_paligemma_q_sarsa_finetune").
    fine_tune_config: str | None = None
    # Number of policy samples to draw per inference call (1 ≤ N ≤ ~16).
    num_samples: int = 8
    # Action selection mode after the critic scores all N samples.
    selection_mode: Literal["argmax", "softmax"] = "argmax"
    # Temperature for softmax selection (ignored when selection_mode="argmax").
    softmax_temperature: float = 1.0
    # If True, takes the elementwise min over the critic ensemble before
    # selecting; matches the LocalPolicy / shirt-hang default.
    take_min_over_ensemble: bool = True
    # Optional override for the slice [offset:offset+critic_action_dim] of the
    # policy action that the critic consumes. None → use the policy config's
    # action_dim_offset.
    critic_action_dim_offset: int | None = None
    # If True, expect the client to populate `obs["critic_image"]` with a
    # critic-pipeline image dict on every infer call. They are routed into the
    # critic's `Observation.images` (the policy still uses `obs["image"]`).
    expect_critic_images: bool = False
    # If True, switch BestOfNPolicy to sample-parallel mode: build a
    # (num_samples, device_count // num_samples) mesh so each candidate runs on
    # its own batch position with a small per-sample FSDP group, instead of
    # sampling device_count candidates and discarding the surplus. Requires
    # device_count divisible by num_samples.
    sample_parallel: bool = False
    # Override the inference mesh's fsdp axis size when sample_parallel=True.
    # None → auto-pick (device_count // num_samples). Use this when the
    # auto-picked value would make per-chip param storage too large
    # (e.g. on v5e-32 keep fsdp=16 to match the saved sharding).
    fsdp_devices: int | None = None
    # If True, BestOfN samples ONE base action from the policy and N different
    # zero-mean Gaussian noise vectors (each with its own rng); the candidates
    # are `action + noise_level * eps_i`. The critic then scores and argmax-
    # selects, same as the standard BestOfN path. Default False reproduces the
    # original "sample N actions from the policy" behavior bit-for-bit.
    inject_noise: bool = False
    # Stddev multiplier on the Gaussian eps when inject_noise=True. Applied in
    # policy-normalized space; for quantile-norm critics the action range is
    # roughly [-1.25, 1.25], so 0.01 is a ~1%-of-range perturbation.
    noise_level: float = 0.0
    # For a predict_subtask_ar critic only: re-decode the current subtask once
    # every N infer calls and reuse the cached subtask string in between (the
    # subtask is a slow-changing phase label). Ignored for non-subtask_ar critics.
    subtask_decode_every: int = 20
    # For a predict_subtask_ar critic only: condition the POLICY prompt on the
    # critic's decoded subtask server-side (fresh on decode-cadence calls in
    # that same call, cached latest in between). When False (default) the
    # policy uses the client-sent prompt unchanged.
    policy_use_decoded_subtask: bool = False


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Override obs["prompt"] for the policy only, leaving the critic on the client-sent
    # prompt — required when the policy was trained with prompt_mode="task_description"
    # but the critic was trained with prompt_mode="subtask". BestOfN path only.
    task_description: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Optional critic config for BestOfN action selection. Default behavior
    # (no critic) is unchanged. Only valid in combination with `policy:checkpoint`.
    critic: CriticArgs | None = None

    # Force routing through BestOfNPolicy even when no critic args are given.
    # Required for RoboCasa-style policies (bimanual_eef_layout, action_dim 32
    # padded down to 14): the BestOfNPolicy infer path applies the manual
    # action-dim slice before Unnormalize, which the default Policy.infer
    # path does not. Implied (no need to set) when --critic.* is set.
    use_bestofn_loader: bool = False
    # Build a (num_samples, device_count // num_samples) mesh for the BC +
    # use_bestofn_loader path (the critic path has its own --critic.sample-parallel).
    # Default True — matches the critic path's typical setting.
    sample_parallel: bool = True
    # Override the inference mesh's fsdp axis size for the BC + use_bestofn_loader
    # path (the critic path has its own --critic.fsdp-devices). Match the value
    # the policy was trained with when the saved sharding doesn't load on the
    # auto-picked mesh (e.g. fsdp=16 for a v5e-32 / target-composite policy).
    fsdp_devices: int | None = None
    # Add zero-mean Gaussian noise to the policy's sampled action in the
    # no-critic (BC) path: action_out = policy_action + noise_level * eps.
    # Applied in policy-normalized space. Default False reproduces the
    # plain BC path bit-for-bit. (For the BoN path, set the equivalent
    # --critic.inject-noise / --critic.noise-level instead.)
    inject_noise: bool = False
    noise_level: float = 0.0

    # Number of flow-matching integration (Euler) steps for the policy's
    # sample_actions. None → use the model default (10). Forwarded into the
    # policy's sample_kwargs; only affects the policy-only (BC) sampling path.
    num_steps: int | None = None


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.BasePolicy:
    """Create a policy from the given arguments."""
    if args.critic is not None or args.use_bestofn_loader:
        # BestOfN path: requires a Checkpoint policy spec (not Default), since
        # the BestOfN loader needs an explicit policy config + checkpoint dir.
        if not isinstance(args.policy, Checkpoint):
            raise ValueError(
                "--critic.* / --use-bestofn-loader requires "
                "`policy:checkpoint --policy.config <name> --policy.dir <dir>`. "
                "Default-environment policies cannot be combined with the BestOfN loader."
            )
        # Lazy import: BestOfNPolicy pulls in JAX value-function modules that
        # the policy-only default path doesn't need.
        from openpi.policies.best_of_n_policy import create_bestofn_policy

        if args.critic is not None:
            return create_bestofn_policy(
                policy_config_name = args.policy.config,
                policy_checkpoint_dir = args.policy.dir,
                policy_step = args.policy.step,
                policy_fine_tune_config = args.policy.fine_tune_config,
                policy_task_description = args.task_description,
                critic_config_name = args.critic.config,
                critic_checkpoint_dir = args.critic.dir,
                critic_step = args.critic.step,
                critic_fine_tune_config = args.critic.fine_tune_config,
                num_samples = args.critic.num_samples,
                take_min_over_ensemble = args.critic.take_min_over_ensemble,
                selection_mode = args.critic.selection_mode,
                softmax_temperature = args.critic.softmax_temperature,
                critic_action_dim_offset = args.critic.critic_action_dim_offset,
                expect_critic_images = args.critic.expect_critic_images,
                default_prompt = args.default_prompt,
                sample_parallel = args.critic.sample_parallel,
                fsdp_devices = args.critic.fsdp_devices,
                inject_noise = args.critic.inject_noise,
                noise_level = args.critic.noise_level,
                subtask_decode_every = args.critic.subtask_decode_every,
                policy_use_decoded_subtask = args.critic.policy_use_decoded_subtask,
                num_steps = args.num_steps,
            )
        # use_bestofn_loader=True without critic: load policy through
        # BestOfNPolicy (which applies the bimanual-EEF action-dim slice)
        # but skip the critic + BestOfN sampling — infer() falls back to the
        # plain policy sampling path internally.
        return create_bestofn_policy(
            policy_config_name = args.policy.config,
            policy_checkpoint_dir = args.policy.dir,
            policy_step = args.policy.step,
            policy_fine_tune_config = args.policy.fine_tune_config,
            policy_task_description = args.task_description,
            default_prompt = args.default_prompt,
            sample_parallel = args.sample_parallel,
            fsdp_devices = args.fsdp_devices,
            inject_noise = args.inject_noise,
            noise_level = args.noise_level,
            num_steps = args.num_steps,
        )

    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
                sample_kwargs={"num_steps": args.num_steps} if args.num_steps is not None else None,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    # Multi-host JAX init for TPU pods. Must run before any other JAX call;
    # load_policy reads jax.local_device_count() and would otherwise hang on
    # multi-host pods waiting for peers. The mapping from JAX process_index
    # to TPU worker IP is deterministic per pod — log it on every worker so
    # you can read off which IP corresponds to rank 0 and point
    # run_eval_robocasa.sh at it.
    if os.environ.get("PLATFORM", "gpu") == "tpu":
        import jax

        jax.distributed.initialize()
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info(
            "JAX distributed: process_index=%d/%d, devices=%d (local=%d), host=%s, ip=%s",
            jax.process_index(), jax.process_count(),
            jax.device_count(), jax.local_device_count(),
            hostname, local_ip,
        )

    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # Multi-host serving split. JAX rank 0 binds the websocket; every other
    # rank enters BestOfNPolicy.participate_loop so the JIT'd inference
    # function (compiled across all hosts) doesn't deadlock on rank 0.
    import jax

    if jax.process_count() > 1 and jax.process_index() != 0:
        from openpi.policies.best_of_n_policy import BestOfNPolicy

        if isinstance(policy, BestOfNPolicy):
            logging.info(
                "JAX rank %d/%d: entering inference participation loop "
                "(rank 0 binds the websocket).",
                jax.process_index(), jax.process_count(),
            )
            policy.participate_loop()
        else:
            logging.warning(
                "JAX rank %d/%d: standard Policy doesn't support multi-host "
                "inference; sleeping. Rank 0 will hang waiting for collectives "
                "unless you launch with --use-bestofn-loader.",
                jax.process_index(), jax.process_count(),
            )
            while True:
                time.sleep(3600)
        return

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
