"""TPU pod configuration for batch_value_learning training.

This file is used by tpc (TPU Pod Controller) to configure and launch
training jobs on TPU pods.
"""

import os
import re


def get_tpu_config(zone, tpu_type, num_tpus, pod_name = None):
    """Generate TPU configuration for a given zone and TPU type."""
    NFS_DIRS = {
        "europe-west4-b": "/nfs/aidm_nfs/saksham",
    }
    CHECKPOINT_DIRS = {
        "europe-west4-b": "gs://saksham-euw4/checkpoints/robocoin/value_functions",
    }
    DATASET_DIRS = {
        "europe-west4-b": "gs://saksham-euw4",
    }

    SOURCE_DIR_NAME = "batch_value_learning"

    nfs = NFS_DIRS[zone]
    if pod_name is not None and pod_name.endswith("-1"):
        nfs = nfs + "3"

    checkpoints_dir = CHECKPOINT_DIRS[zone]
    dataset_dir = DATASET_DIRS[zone]

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
        "train_args": {
            "batch-size": 256,
            "checkpoint-base-dir": checkpoints_dir,
            "data.tfds-data-dir": dataset_dir,
            "project-name": "robocoin_value_learning",
        },
    }


DEFAULT_TRAIN_ARGS = {
    "log-interval": 100,
}

TPU_POD_CONFIGS = {
    "eu-v5-64-0": get_tpu_config("europe-west4-b", "v5", 64),
    "eu-v5-64-1": get_tpu_config("europe-west4-b", "v5", 64, "v5e-tpu-64-1"),
    "eu-v5-128": get_tpu_config("europe-west4-b", "v5", 128),
    "eu-v5-256": get_tpu_config("europe-west4-b", "v5", 256),
}

TPU_POD_TYPES = {
    "v5e-tpu-256*": "eu-v5-256",
    "v5e-tpu-128*": "eu-v5-128",
    "v5e-tpu-64-0": "eu-v5-64-0",
    "v5e-tpu-64-1": "eu-v5-64-1",
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

train_args = os.environ.get("TRAIN_ARGS", "")
config_name = os.environ.get("CONFIG_NAME", "robocoin_paligemma_v_mc")
train_args = DEFAULT_TRAIN_ARGS | config["train_args"] | parse_args(train_args)

# Append Q or V suffix to checkpoint directory based on config name
suffix = "Q" if "_q_" in config_name.lower() else "V"
if "checkpoint-base-dir" in train_args:
    train_args["checkpoint-base-dir"] = f"{train_args['checkpoint-base-dir']}/{suffix}"

train_args_str = " \\\n\t".join([f"--{k}={v}" for k, v in train_args.items()])
train_script = os.environ.get("TRAIN_SCRIPT", "scripts/train_value_function.py")
# train_script = "scripts/train_value_function_debug.py"
NFS_USER = os.environ.get("NFS_USER", "saksham")
CHOWN_USER = os.environ.get("CHOWN_USER", NFS_USER)

launch_script = f"""
#!/bin/bash

echo "Running cd {config["code_dir"]}"
cd {config["code_dir"]}

echo "TRAIN_SCRIPT: {train_script}"
echo "CONFIG_NAME: {config_name}"
echo "TRAIN_ARGS: {train_args_str}"
echo "SRC_DIR: {config["src_dir"]}"
echo "NFS_USER: {NFS_USER}"
echo "CHOWN_USER: {CHOWN_USER}"

echo "Running source {config["src_dir"]}/vla/bin/activate"
source {config["src_dir"]}/vla/bin/activate

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
export LD_LIBRARY_PATH="/nfs/aidm_nfs/{NFS_USER}/ffmpeg-7/lib:"

echo "WANDB_API_KEY: $WANDB_API_KEY"
export WANDB_API_KEY="<wandb_api_key>"

echo "Running sudo chown -R {CHOWN_USER}:{CHOWN_USER} /tmp/tpu_logs"
sudo chown -R {CHOWN_USER}:{CHOWN_USER} /tmp/tpu_logs

echo "Running python {train_script} {config_name} {train_args_str} --resume"
python {train_script} {config_name} \\
    {train_args_str} --resume
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
