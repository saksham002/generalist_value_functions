# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

openpi is Physical Intelligence's open-source repository for Vision-Language-Action (VLA) models for robotics. It contains:
- **π₀**: Flow-based VLA model
- **π₀-FAST**: Autoregressive VLA with FAST action tokenizer
- **π₀.₅**: Upgraded π₀ with better generalization via knowledge insulation
- **Value Functions**: MLP-based value function models for RL (recent addition)

The repository supports both JAX and PyTorch implementations, with JAX being the primary framework and PyTorch support validated on LIBERO benchmark.

## Development Commands

### Environment Setup
```bash
# Install dependencies (uses uv package manager)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Update submodules (required for some examples)
git submodule update --init --recursive
```

### Testing
```bash
# Run all non-manual tests
uv run pytest --strict-markers -m "not manual"

# Run specific test file
uv run pytest src/openpi/models/model_test.py

# Run single test
uv run pytest src/openpi/models/model_test.py::test_name
```

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

## Code Style and Quality Conventions

### General Code Style
- **Line length**: 120 characters maximum
- **Target Python version**: 3.11
- **Import style**: Force single-line imports (enforced by isort), except for `collections.abc`, `typing`, and `typing_extensions`
- **Type hints**: Use type hints extensively, especially with `array_typing` module for array shapes
- **Dataclasses**: Prefer frozen dataclasses for configuration objects
- **Variable names**: Use descriptive variable names that clearly convey purpose (e.g., `num_transitions_per_sample` instead of `n` or `num_transitions`), don't worry about variable names being too long (up to reasonable length).
- **Avoid reshape with -1**: Don't use `reshape(..., -1)` as it can lead to silent bugs. Instead, explicitly compute and specify all dimensions.

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

### Running on Cluster
If the user asks to run on the cluster (e.g., for GPU access, legacy D4RL with mujoco-py, etc.):

1. **Sync local changes to the cluster** (run this locally first):
   ```bash
   rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude 'wandb' \
       /Users/maxsobolmark/dev/batch_value_learning/ babel:~/dev/batch_value_learning/
   ```

2. **Run training via srun** (directly from local machine):
   ```bash
   ssh babel "srun -p debug --mem=64GB --gres=gpu:L40S:1 --time=2:00:00 bash -c 'source /home/jsobolma/bashrc_max && export PATH=\$HOME/.local/bin:\$PATH && export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/jsobolma/.mujoco/mujoco210/bin:/usr/lib/nvidia && cd ~/dev/batch_value_learning && uv run python scripts/train_value_function.py YOUR_CONFIG --num_train_steps 100 --no-wandb_enabled'"
   ```

3. **The code is located at**: `~/dev/batch_value_learning`

4. **Legacy D4RL Setup**:
   - The cluster has MuJoCo 2.1 installed at `~/.mujoco/mujoco210`
   - Need to install `cython<3` for mujoco-py compatibility: `uv pip install "cython<3"`
   - Install d4rl dependencies: `uv pip install gym==0.23.1 && uv pip install "d4rl @ git+https://github.com/Farama-Foundation/D4RL.git" --no-deps && uv pip install "mujoco-py<2.2,>=2.1"`
   - A pre-compiled mujoco_py .so file exists in the `parl` conda env and can be copied if build fails
