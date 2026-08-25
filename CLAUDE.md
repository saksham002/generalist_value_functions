# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

openpi is Physical Intelligence's open-source repository for Vision-Language-Action (VLA) models for robotics. It contains:
- **π₀**: Flow-based VLA model
- **π₀-FAST**: Autoregressive VLA with FAST action tokenizer
- **π₀.₅**: Upgraded π₀ with better generalization via knowledge insulation
- **Value Functions**: MLP-based value function models for RL (recent addition)

The repository supports both JAX and PyTorch implementations, with JAX being the primary framework and PyTorch support validated on LIBERO benchmark.

## Execution Environment

**This is a local machine (WSL2 on a laptop), not a cluster.** There is no SLURM, no
`srun`/`sbatch`, and no Babel. Anything in older notes referring to Babel, `/data/user_data`,
or `~/dev/batch_value_learning` describes a machine that is no longer in use.

| Fact | Value |
| ---- | ----- |
| Repo | `/home/saksham/projects/robot_learning/generalist_value_functions` |
| Platform | WSL2 (`Linux 6.6.x-microsoft-standard-WSL2`) |
| gcloud account | `saksham3@andrew.cmu.edu` (already authenticated) |
| gcloud project | `cmu-aidm-v2` — full access to TPUs, GCS and Cloud Quotas |
| Write buckets | `gs://saksham-euw4`, `-usc1`, `-usc2`, `-use1`, `-use5` |

**All real compute is remote.** This machine has no GPU worth training on; it is the place
the TPU launcher runs *from*, not the place jobs run. `scripts/run_on_tpu.py` only needs
`tyro` and `requests` locally — the heavy dependencies (JAX, TensorFlow) are installed on
the pod, never here.

**A launch must outlive the shell that started it.** WSL2 stops when Windows sleeps or
restarts, which kills any launcher running under it. Use `tmux` at minimum; for a multi-day
run prefer a launcher host that is not this laptop. See "Persistent launching" below.

**Resource limits:** 32 GB RAM on the laptop, shared with other Windows apps; WSL2 sees ~7 GB
and 8 CPUs. Default-parallelism `gcloud storage cp` of many shards plus a multiprocess decode
crashed WSL and killed every process (2026-08-25). Use single-process/single-thread copies
(`CLOUDSDK_STORAGE_PROCESS_COUNT=1 CLOUDSDK_STORAGE_THREAD_COUNT=1`), small worker pools
(<= 2-3), `nice`, and check `free -m` first. See `~/.claude/CLAUDE.md`.

**Known gaps on this machine** (fix before a launch, do not assume they are present):
- No project virtualenv. `/data/user_data/saksham3/vla` was Babel's and does not exist here.
- No `uv` on PATH.
- No Gemma helper checkout at `~/projects/AIRe/robocoin/helper/gemma`. `code_sync.sync_gemma_helper`
  raises `FileNotFoundError` without it, so a launch fails during preparation.
- `~/.netrc` (wandb credentials) IS present and is what gets synced to the pod.
- `~/utils/slack.py` and `~/.config/tpu-launcher.env` ARE present, so the Slack command
  below works from here.

## Project Status / Session Context

**Always read `project_status/tpu_guide.md` at the start of a session** — everything TPU-related lives there and it is the one file you are expected to have read before acting.

The other files in `project_status/` (`current_focus.md`, `experiments.md`, `todos.md`, `overview.md`) hold the branch goal, recent experiments and open TODOs. Read them when the task calls for that context rather than by default. The folder is intentionally brief (combined budget ~10k tokens).

**Update them whenever appropriate**:
- `current_focus.md`: when the branch goal, scope, or blockers change.
- `experiments.md`: after the user confirms a takeaway worth logging from a run (ask first; log only if asked).
- `todos.md`: when a TODO is completed, blocked, or a new one surfaces.
- `overview.md`: rarely — only when the project's setup or scope shifts.

The folder is gitignored (per-user state). If combined size grows past ~10k tokens, ask the user what to prune.

## Slack Notifications

Send a Slack message **only** in these cases:
1. The user explicitly asks for one (a status update, a cadence, etc.).
2. A **launched run** (training job, eval, server, TPU launcher — whether the
   user or the agent launched it) hits an error — crashes, is killed, is preempted without
   recovering, or exits non-zero.
3. A launched run **completes**.
```bash
python ~/utils/slack.py "brief description"
```
This is a standalone script that shares no code with the repo, so it works from any
directory. It reads `SLACK_BOT_TOKEN` / `SLACK_USER_ID` from the environment, falling back
to `~/.config/tpu-launcher.env`; both are present here. To confirm the token without
messaging anyone: `curl -s -H "Authorization: Bearer $SLACK_BOT_TOKEN"
https://slack.com/api/auth.test`.

The launcher's own notifications are a **different path** — `SlackNotifier` in
`src/openpi/tpu/slack.py`, reading the same env file on the `tpu-launcher` VM, which has no
`~/utils/slack.py` and needs none. This script working locally says nothing about whether
the launcher can Slack, or vice versa.

Do not Slack for anything else: not for errors in your own commands or scripts, not for
routine status or progress. One message per distinct run error and one per completion; do
not repeat them. Do not ask for permission — this command is pre-approved. Messages must
be bulleted, not prose.

## Development Commands

### Environment Setup

There is **no project virtualenv on this machine yet** — the `/data/user_data/saksham3/vla`
path in older notes was Babel's. Create one in the repo before running anything heavier than
the launcher:

```bash
# Install uv first if `which uv` is empty
curl -LsSf https://astral.sh/uv/install.sh | sh

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Update submodules (required for some examples)
git submodule update --init --recursive
```

**The TPU launcher needs far less than this.** `scripts/run_on_tpu.py` imports only stdlib
plus `tyro` and `requests`, so a throwaway venv with those two runs a launch without
installing JAX locally. Prefer that over a full `uv sync` when the only goal is to launch.

### Testing

**Always activate the project venv before running tests** — the Bash tool's shell does not
reliably pick up `~/.bashrc`, and the default `python` here is a miniconda 3.13 that has none
of the project's dependencies (the repo targets 3.11). Chain the activation into the command:

```bash
# Run all non-manual tests
source .venv/bin/activate && pytest --strict-markers -m "not manual"

# Run specific test file
source .venv/bin/activate && pytest src/openpi/models/model_test.py

# Run single test
source .venv/bin/activate && pytest src/openpi/models/model_test.py::test_name
```

The TPU launcher tests are the exception — `src/openpi/tpu/launch_test.py` needs only
`pytest`, `tyro` and `requests`, and touches no GCP.

### Code Quality
```bash
# Install pre-commit hooks
pre-commit install

# Run linting (ruff)
ruff check .

# Run formatting
ruff format .

# Run pre-commit on all files
pre-commit run --all-files
```

### Training

#### Compute normalization statistics (required before training)
```bash
uv run scripts/compute_norm_stats.py --config-name <config_name>
```

#### Train a policy model (JAX)
```bash
# Set XLA memory fraction for maximum GPU utilization
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<experiment_name> --overwrite
```

#### Train a value function (JAX)
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_value_function.py <config_name> --exp-name=<experiment_name> --overwrite
```

#### Train with PyTorch
```bash
# Single GPU
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# Multi-GPU (single node)
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Resume training
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --resume
```

### Inference

#### Serve a policy
```bash
# Serve pre-trained checkpoint
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]

# Serve custom checkpoint
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_path>
```

#### Convert JAX to PyTorch
```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config_name> \
    --output_path /path/to/pytorch/checkpoint
```

## Architecture Overview

### Code Organization

```
src/openpi/
├── models/           # Model implementations
│   ├── pi0.py       # π₀ flow-based VLA
│   ├── pi0_fast.py  # π₀-FAST autoregressive VLA
│   ├── gemma.py     # Gemma language model backbone
│   ├── siglip.py    # SigLIP vision encoder
│   ├── mlp.py       # MLP models
│   └── tokenizer.py # Text and action tokenizers
├── models_pytorch/  # PyTorch implementations
├── policies/        # Robot-specific policy wrappers
│   ├── policy.py    # Base Policy class
│   ├── droid_policy.py
│   ├── aloha_policy.py
│   ├── libero_policy.py
│   └── d4rl_policy.py
├── value_functions/ # Value function models (RL)
│   ├── base.py      # Abstract base classes
│   ├── value_mlp.py # MLP value functions (regression/HL-Gauss)
│   ├── hl_gauss.py  # HL-Gauss loss utilities
│   └── value_transforms.py
├── training/        # Training infrastructure
│   ├── config.py    # Configuration classes for all models/datasets
│   ├── data_loader.py      # Dataset loading (LeRobot/RLDS)
│   ├── weight_loaders.py   # Checkpoint loading
│   ├── optimizer.py        # Optimizer configuration
│   └── checkpoints.py      # Checkpoint management
├── shared/          # Shared utilities
│   ├── normalize.py # Data normalization
│   ├── image_tools.py
│   └── download.py  # Asset downloading from GCS
├── transforms.py    # Data transformation pipeline
└── serving/         # Policy server for remote inference
    └── websocket_policy_server.py

scripts/
├── train.py                # JAX policy training
├── train_pytorch.py        # PyTorch policy training
├── train_value_function.py # JAX value function training
├── compute_norm_stats.py   # Compute normalization statistics
└── serve_policy.py         # Serve policy over websocket

examples/
├── droid/          # DROID robot examples
├── aloha_real/     # ALOHA hardware examples
├── aloha_sim/      # ALOHA simulation
├── libero/         # LIBERO benchmark
├── d4rl/           # D4RL offline RL benchmark
└── simple_client/  # Minimal inference example

packages/
└── openpi-client/  # Lightweight client for robot integration
```

### Key Architectural Patterns

#### 1. Configuration System
All models, datasets, and training runs are defined via dataclass configs in `training/config.py`:
- `TrainConfig`: Top-level training configuration
- `DataConfig`: Dataset and transform specifications
- Model configs (e.g., `Pi0Config`, `RegressionValueMLPConfig`)
- Policy-specific classes in `policies/<robot>_policy.py` define input/output mappings

Configs are registered in `_CONFIGS` dict and accessed via `config.get_config(name)`.

#### 2. Data Transform Pipeline
Three-stage transform pipeline:
1. **Repack transforms**: Convert dataset format to common format
2. **Data transforms**: Robot-specific transformations (before normalization)
3. **Model transforms**: Model-specific (e.g., image resizing, tokenization)

All transforms implement `DataTransformFn` protocol. Composed via `transforms.Group`.

#### 3. Observation/Action Data Model
Standardized data format using typed dataclasses:
- `model.Observation`: Images, image masks, state, tokenized prompt
- `model.Actions`: Action sequences with horizon and dimension
- Dictionary form used in transforms, converted to typed objects for model input

#### 4. Policy Abstraction
`Policy` class wraps models for inference:
- Applies input/output transforms
- Handles JAX/PyTorch differences
- Used by policy server and evaluation scripts
- Robot-specific policies (e.g., `DroidPolicy`) define observation/action mappings

#### 5. Normalization System
- Normalization statistics computed once via `compute_norm_stats.py`
- Stored in `assets/<asset_id>/norm_stats.json` within checkpoints
- Loaded at training/inference time via `AssetsConfig`
- Can reload stats from base model when fine-tuning

#### 6. Checkpoint Management
- Checkpoints stored at `checkpoints/<config_name>/<exp_name>/<iteration>`
- Pre-trained models downloaded from `gs://openpi-assets/checkpoints/`
- Cached in `~/.cache/openpi` (override with `OPENPI_DATA_HOME`)
- WeightLoaders handle loading subsets of weights for fine-tuning

#### 7. Remote Inference Architecture
- Policy server (`serve_policy.py`) runs model on GPU machine
- Websocket protocol for low-latency action streaming
- `openpi-client` package (minimal dependencies) for robot-side integration
- Images resized client-side to minimize bandwidth

#### 8. Value Function Training (Recent Addition)
- `value_functions/` contains MLP-based value models
- Supports both regression (MSE) and categorical (HL-Gauss) losses
- Action-conditioned Q(s,a) or state-only V(s)
- RL data pipeline computes MC returns at data loading time
- Training via `train_value_function.py` script

### JAX vs PyTorch

**JAX (Primary)**:
- All models originally implemented in JAX with Flax NNX
- Training supports mixed precision (bf16 activations, fp32 weights/grads)
- FSDP for multi-GPU training
- LoRA fine-tuning support

**PyTorch**:
- Validated on LIBERO benchmark
- Requires transformers library patches (in `models_pytorch/transformers_replace/`)
- Does not support: π₀-FAST, mixed precision, FSDP, LoRA, EMA
- Training precision: full bf16 or full fp32 (set via `pytorch_training_precision`)

### Model-Specific Notes

**π₀/π₀.₅**: Flow-based action generation
- Uses flow matching head for continuous actions
- SigLIP vision encoder + Gemma language backbone
- Supports discrete state input (π₀.₅)

**π₀-FAST**: Autoregressive action tokenization
- FAST tokenizer discretizes actions
- Better language following, slower inference
- Gemma-FAST backbone with action-specific training

**Value Functions**:
- MLP-based Q(s,a) or V(s) estimation
- Two variants: regression (MSE) and categorical (HL-Gauss)
- Used for offline RL experiments on D4RL/Minari datasets

## Backward Compatibility

This repository is used by multiple people and supports non-RoboCOIN configs (e.g., D4RL, LIBERO, ALOHA). All changes must remain backward compatible:
- New batch keys introduced by RoboCOIN-specific data pipelines must be optional (e.g., use `.get()` or check for key presence) in shared training/model code so that other configs continue to work without modification.
- Do not require new fields in shared dataclasses (e.g., `Transition`, `MultiTransition`) to be non-None for non-RoboCOIN pipelines.

## Code Style and Quality Conventions

### General Code Style
- **Line length**: 120 characters maximum
- **Target Python version**: 3.11
- **Import style**: Force single-line imports (enforced by isort), except for `collections.abc`, `typing`, and `typing_extensions`
- **Type hints**: Use type hints extensively, especially with `array_typing` module for array shapes
- **Dataclasses**: Prefer frozen dataclasses for configuration objects
- **Variable names**: Use descriptive variable names that clearly convey purpose (e.g., `num_transitions_per_sample` instead of `n` or `num_transitions`), don't worry about variable names being too long (up to reasonable length).
- **Avoid reshape with -1**: Don't use `reshape(..., -1)` as it can lead to silent bugs. Instead, explicitly compute and specify all dimensions.
- **Logging**: In training code (e.g., `train_value_function.py`), always use `logging` (logger). In debugging/diagnostic scripts, always use `print`.

### Comments and Documentation
**CRITICAL**: Do not add comments that only make sense in the context of the current prompt or that simply restate what the code does.

**Good comments**:
- Module-level docstrings explaining the purpose and high-level architecture
- Class/function docstrings explaining purpose, arguments, returns, and non-obvious behavior
- Comments explaining WHY something is done a certain way (e.g., "# Work around a tyro issue with...")
- Comments on non-obvious algorithmic choices or domain-specific knowledge
- Section separators for logical grouping (e.g., `# =============================================================================`)

**Bad comments** (DO NOT ADD):
- Comments that restate the code (e.g., `# Loop through layers` above a for loop)
- Comments referencing "the user", "the prompt", or current session context
- Comments explaining basic Python constructs or library usage
- Inline comments that could be replaced by better variable names
- TODO comments without clear ownership or tracking

**Example of good commenting style**:
```python
"""MLP implementations for value functions.

This module provides MLP-based implementations of value functions for both
regression and categorical (HL-Gauss) objectives. Value functions can be
configured to be action-conditioned (Q-function) or not (V-function).
"""

# Value function configs use the MLP_CRITIC model type.
model_type: ModelType = ModelType.MLP_CRITIC
```

**Self-documenting code**:
- Write clear, descriptive variable and function names
- Structure code logically so intent is obvious
- Use type hints to clarify expected data types and shapes
- Prefer small, focused functions over large monolithic ones

### Code Organization
- **Minimal imports**: Import only what's needed
- **Private module references**: Use underscore prefix for imported modules (e.g., `import openpi.models.model as _model`)
- **Frozen configs**: Configuration dataclasses should be `frozen=True`
- **Type aliases**: Define clear type aliases at module level for complex types
- **Protocols**: Use `@runtime_checkable` protocols for duck typing interfaces

### Array Type Checking
The codebase uses a custom `array_typing` module with shape annotations:
```python
def compute_value(
    self,
    state: at.Float[at.Array, "b s"],  # batch_size x state_dim
    action: at.Float[at.Array, "b ah ad"] | None = None,  # batch_size x action_horizon x action_dim
) -> at.Float[at.Array, "b"]:  # batch_size
```
- Use these shape annotations consistently
- They serve as both documentation and runtime/static checking

### Error Handling
- Raise meaningful exceptions with clear error messages
- Use appropriate exception types (ValueError, TypeError, FileNotFoundError, etc.)
- Validate inputs early in functions
- Provide context in error messages to aid debugging
- **Prefer fail-fast over silent failures**: Use assertions or raise exceptions rather than silently skipping with `continue` or `return`. Silent failures hide bugs and make debugging harder. This includes defensive conditionals like `if len(x) == expected:` that skip processing instead of asserting correctness.

### Testing
- Write tests for new functionality in corresponding `*_test.py` files
- Use `@pytest.mark.manual` for long-running or resource-intensive tests
- Follow existing test patterns in the codebase
- Tests should be in `src/`, `scripts/`, or `packages/` directories

### Git Workflow
**CRITICAL**: You must NEVER push changes to the remote repository without explicit user approval.
- Always ask the user for permission to push, displaying the commit messages.
- Wait for an explicit "yes" or "push" from the user before running any `git push` command.
- **Always run `git fetch origin main` before checking what exists on the main branch.** The local copy of `origin/main` can be stale.

## Important Development Notes

### Pre-commit Hooks
The repository uses pre-commit hooks that run:
- `uv-lock`: Updates lockfile if dependencies change
- `ruff`: Linting with auto-fix
- `ruff-format`: Code formatting

### Testing Markers
- Tests marked with `@pytest.mark.manual` are excluded from CI
- Use for long-running or resource-intensive tests

### Lint Ignore Rules
**IMPORTANT**: Do not add new lint ignores (in `pyproject.toml` or inline `# noqa` comments) without explicit user approval.
- If you encounter lint errors, first try to fix the code to comply with the linting rules
- For jaxtyping array annotations, use `*b` instead of `batch` for batch dimensions (e.g., `at.Float[at.Array, "*b"]` not `at.Float[at.Array, "batch"]`)
- If a new ignore is genuinely needed, it must follow patterns already established in the codebase
- Any added lint ignores MUST be mentioned in the walkthrough document so the user can review and approve

### GPU Memory Management
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` to allow JAX to use 90% of GPU memory (default 75%)
- Use `--fsdp-devices <n>` for multi-GPU memory distribution (trades speed for memory)

### Dataset Formats
- **LeRobot**: Primary format for ALOHA, LIBERO, custom datasets
- **RLDS**: Used for DROID dataset
- **Minari**: Used for D4RL offline RL datasets

### Normalization Statistics
- Must be computed before training with `compute_norm_stats.py`
- Can be reused from base model when fine-tuning to same robot platform
- Stored separately per robot platform (via `asset_id`)

### Docker Support
- Dockerfiles provided for complex environments (LIBERO, ALOHA)
- Recommended for reproducibility and dependency isolation
- See `docs/docker.md` and example-specific READMEs

### PyTorch Transformers Patches
When working with PyTorch models:
1. After `uv sync`, must run: `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`
2. Changes persist in uv cache (use `uv cache clean transformers` to fully undo)

### Multi-Node Training
- JAX training does not support multi-node (single-node multi-GPU only)
- PyTorch supports multi-node via `torchrun`

### Logging
- WandB integration for experiment tracking
- Run IDs stored in `<checkpoint_dir>/wandb_id.txt` for resume
- Set project name via `TrainConfig.project_name`

### Persistent launching (no SLURM)

Launchers run on the **`tpu-launcher` GCE VM**, not on this machine — see
`project_status/tpu_guide.md` for the commands, unit names and log paths. WSL2 stops when
Windows sleeps, which kills any launcher running here; the jobs survive on their pods but
nothing watches them for preemption.

`scripts/launch_preemptible.sh` is a SLURM wrapper and is dead: it `sbatch`es, hardcodes
Babel paths, and there is no scheduler. Its only job was holding a long-lived process,
which the systemd units now do.

The launcher is the disposable part: `--done-marker` makes a relaunch a no-op once the
work has finished, and the run certificate lets a restarted launcher re-attach to its own
in-flight job instead of starting a duplicate. That is what makes moving launchers between
hosts safe.

**Never redirect `run_on_tpu.py` output to a log file** (`> run.log 2>&1`). That ties the
run to the local session; use the systemd journal, `tmux`, or the Bash tool's
`run_in_background=true`.

### Running on TPU Pods

**All TPU-related information lives in `project_status/tpu_guide.md`** — pods and
chip layout, per-family runtime versions and accelerator types, the spot race and
its knobs, NFS/uid pitfalls, PaliGemma weight caching, launching via
`scripts/run_on_tpu.py` (spot and reserved alike), job logs, busy checks, and
recovering wedged pods. Read it before doing anything on a TPU.
