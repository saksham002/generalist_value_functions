"""TPU pod configuration for batch_value_learning training.

This file is used by tpc (TPU Pod Controller) to configure and launch
training jobs on TPU pods.
"""

import os
import re


def get_tpu_config(zone, tpu_type, num_tpus):
    """Generate TPU configuration for a given zone and TPU type."""
    NFS_DIRS = {
        "europe-west4-b": "/nfs/aidm_nfs/saksham3",
    }
    CHECKPOINT_DIRS = {
        "europe-west4-b": "gs://saksham-euw4/checkpoints/robocoin/value_functions",
    }
    DATASET_DIRS = {
        "europe-west4-b": "gs://saksham-euw4",
    }

    SOURCE_DIR_NAME = "batch_value_learning"

    nfs = NFS_DIRS[zone]
    checkpoints_dir = CHECKPOINT_DIRS[zone]

    runtime_versions = {"v4": "tpu-ubuntu2204-base", "v5": "v2-alpha-tpuv5-lite"}
    accelerator_types = {"v4": f"v4-{num_tpus}", "v5": f"v5litepod-{num_tpus}"}

    return {
        "tpc_args": {
            "project": "cmu-aidm-v2",
            "zone": zone,
            "accelerator_type": accelerator_types[tpu_type],
            "runtime_version": runtime_versions[tpu_type],
            "reserved": True,
        },
        "setup_script": "source $HOME/.bashrc",
        "src_dir": f"{nfs}/uv", # fixed
        "code_dir": f"{nfs}/{SOURCE_DIR_NAME}",
        "extra_args": {
            "batch-size": 256,
            "checkpoint-base-dir": checkpoints_dir,
            "project-name": "robocoin_value_learning",
        },
    }


DEFAULT_EXTRA_ARGS = {
    "log-interval": 100,
}

TPU_POD_CONFIGS = {
    "eu-v5e-0": get_tpu_config("europe-west4-b", "v5", 64),
    "eu-v5-64-0": get_tpu_config("europe-west4-b", "v5", 64),
    "eu-v5-64-1": get_tpu_config("europe-west4-b", "v5", 64),
    "eu-v5-128": get_tpu_config("europe-west4-b", "v5", 128),
    "eu-v5-256": get_tpu_config("europe-west4-b", "v5", 256),
}

TPU_POD_TYPES = {
    "v5e-tpu-256*": "eu-v5-256",
    "v5e-tpu-128*": "eu-v5-128",
    "v5e-tpu-64-0": "eu-v5-64-0",
    "v5e-tpu-64-1": "eu-v5-64-1",
    "v5e-0": "eu-v5e-0",
}


def parse_args(args_str):
    """Parse comma-separated key=value arguments."""
    args = {}
    if args_str:
        for arg in args_str.split(","):
            key, value = arg.split("=")
            args[key] = value
    return args


pod_name = os.environ.get("POD_NAME")
for config_re, maybe_pod_type in TPU_POD_TYPES.items():
    if re.match(config_re, pod_name):
        pod_type = maybe_pod_type
        break

config = TPU_POD_CONFIGS[pod_type]

raw_extra_args = os.environ.get("EXTRA_ARGS", "")
config_name = os.environ.get("CONFIG_NAME", "robocoin_paligemma_v_mc")

run_script = os.environ.get("RUN_SCRIPT", "scripts/train_value_function.py")
is_eval = "evaluate_value_function" in run_script


def _format_arg(k, v):
    v_str = str(v)
    if " " in v_str:
        return f"--{k}='{v_str}'"
    return f"--{k}={v_str}"


if is_eval:
    # Evaluation mode: parse eval-specific args from EXTRA_ARGS.
    # Episode format in $4: "." separates repo_idx from ep_idx, "+" separates pairs.
    #   e.g. "episodes=3.270+0.15,cache-dir=/nfs/..." → --episodes 3,270 0,15
    eval_args = parse_args(raw_extra_args)

    if "episodes" not in eval_args:
        raise ValueError("evaluate_value_function requires 'episodes' in EXTRA_ARGS (e.g. episodes=3.270+0.15)")
    if "cache-dir" not in eval_args:
        raise ValueError("evaluate_value_function requires 'cache-dir' in EXTRA_ARGS")

    episodes_raw = eval_args["episodes"]
    episodes_str = " ".join(pair.replace(".", ",") for pair in episodes_raw.split("+"))

    checkpoints_dir = config["extra_args"]["checkpoint-base-dir"]
    suffix = "Q" if "_q_" in config_name.lower() else "V"
    checkpoint_path = f"{checkpoints_dir}/{suffix}/{config_name}/{config_name}"
    model_spec = f"{config_name}:{checkpoint_path}"

    eval_cmd_parts = [
        f"--model {model_spec}",
        f"--episodes {episodes_str}",
        f"--cache-dir {eval_args['cache-dir']}",
    ]
    if "project-name" in eval_args:
        eval_cmd_parts.append(f"--project-name {eval_args['project-name']}")

    run_args_str = " \\\n\t".join(eval_cmd_parts)
else:
    extra_args = DEFAULT_EXTRA_ARGS | config["extra_args"] | parse_args(raw_extra_args)

    # Set checkpoint directory based on training script
    if "train_value_function" in run_script:
        suffix = "Q" if "_q_" in config_name.lower() else "V"
        if "checkpoint-base-dir" in extra_args:
            extra_args["checkpoint-base-dir"] = f"{extra_args['checkpoint-base-dir']}/{suffix}"
        extra_args["wandb-group"] = "Value Functions"
    else:
        extra_args["checkpoint-base-dir"] = "gs://saksham-euw4/checkpoints/robocoin/pi05_finetune"
        extra_args["wandb-group"] = "Policies"

    run_args_str = " \\\n\t".join([_format_arg(k, v) for k, v in extra_args.items()])
# run_script = "scripts/train_value_function_debug.py"
NFS_USER = os.environ.get("NFS_USER", "saksham3")

launch_script = f"""
#!/bin/bash

echo "Running cd {config["code_dir"]}"
cd {config["code_dir"]}

echo "RUN_SCRIPT: {run_script}"
echo "CONFIG_NAME: {config_name}"
echo "EXTRA_ARGS: {run_args_str}"
echo "SRC_DIR: {config["src_dir"]}"
echo "NFS_USER: {NFS_USER}"

echo "Running source {config["src_dir"]}/vla/bin/activate"
source {config["src_dir"]}/vla/bin/activate

if echo "{config_name}" | grep -q "gemma3"; then
    echo "Gemma 3 config: skipping PaliGemma 2B checkpoint copy"
else
    PALIGEMMA_CACHE=$HOME/.cache/openpi/vertex-model-garden-paligemma-us/paligemma/pt_224.npz
    if [ ! -f "$PALIGEMMA_CACHE" ]; then
        echo "PaliGemma checkpoint not found at $PALIGEMMA_CACHE, copying from NFS..."
        mkdir -p "$(dirname "$PALIGEMMA_CACHE")"
        cp /nfs/aidm_nfs/saksham3/gemma/2b/pt_224.npz "$PALIGEMMA_CACHE"
    fi
fi

# Set platform for TPU distributed training

export PLATFORM=tpu
export GCS_READ_CACHE_BLOCK_SIZE_MB=0
export GCS_READ_CACHE_MAX_STALENESS=0


echo "Python: $(python -V)"
echo "Checking TensorFlow runtime..."
python - <<'PY'
from importlib import metadata

def dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

tf_dist = dist_version("tensorflow")
tf_cpu_dist = dist_version("tensorflow-cpu")
print("Dist tensorflow:", tf_dist)
print("Dist tensorflow-cpu:", tf_cpu_dist)

import tensorflow as tf
print("Imported tensorflow:", tf.__version__, "tf.__file__:", tf.__file__)

import os
print("GCS_READ_CACHE_BLOCK_SIZE_MB:", os.environ.get("GCS_READ_CACHE_BLOCK_SIZE_MB", "<not set>"))
print("GCS_READ_CACHE_MAX_STALENESS:", os.environ.get("GCS_READ_CACHE_MAX_STALENESS", "<not set>"))
PY

echo "Running {config["setup_script"]}"
{config["setup_script"]}

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "WANDB_API_KEY: $WANDB_API_KEY"

echo "Running sudo chmod -R 777 /tmp/tpu_logs"
sudo chmod -R 777 /tmp/tpu_logs

if [[ "$(hostname)" == *-w-0 ]]; then
    echo "Worker 0: fixing NFS permissions..."
    sudo chmod -R 777 /nfs/aidm_nfs/saksham3/batch_value_learning/
    sudo chmod -R 777 /nfs/aidm_nfs/saksham3/robocoin
fi

{"" if is_eval else f'echo "Running python {run_script} {config_name} --resume {run_args_str}"'}
{"" if is_eval else f"python {run_script} {config_name} --resume"}{"" if not is_eval else f'echo "Running python {run_script} {run_args_str}"'}
{"" if not is_eval else f"python {run_script}"} \
	{run_args_str}
echo "Script exited with code $?"
sleep 30
EXIT_CODE=$?
echo "Script exited with code $EXIT_CODE"
exec bash
"""

if os.environ.get("VERBOSE", "0") == "1":
    import pprint

    print("*" * 100)
    print("RUNNING WITH CONFIG")
    pprint.pprint(config)
    print("LAUNCH SCRIPT:")
    print(launch_script)
    print("*" * 100)

configure_tpc(
    **config["tpc_args"],
    name=pod_name,
    tmux_session_name=f"tpc_{pod_name}",
    launch_script=launch_script,
)
