# Minari/D4RL Dataset Conversion

This directory contains scripts for converting Minari datasets (the modern successor to D4RL) to LeRobot format.

## Installation

```bash
uv pip install minari
```

## Usage

### Convert a Minari dataset

```bash
# Convert antmaze dataset  
uv run examples/d4rl/convert_d4rl_to_lerobot.py --dataset_id D4RL/antmaze/large-diverse-v1

# List available datasets
python -c "import minari; print([d for d in minari.list_remote_datasets() if 'antmaze' in d])"
```

## Supported Datasets

Any Minari dataset can be converted. D4RL datasets are available under the `D4RL/` namespace:
- **Antmaze**: `D4RL/antmaze/umaze-v1`, `D4RL/antmaze/large-diverse-v1`, etc.
- **Locomotion**: `D4RL/halfcheetah/medium-v2`, `D4RL/hopper/medium-v2`, etc.
- **Adroit**: `D4RL/pen/human-v1`, `D4RL/hammer/human-v1`, etc.

See [Minari documentation](https://minari.farama.org/) for full list.
