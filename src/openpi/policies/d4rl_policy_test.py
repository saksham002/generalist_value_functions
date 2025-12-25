import numpy as np

from openpi.policies import d4rl_policy


def test_d4rl_inputs():
    """Test D4RLInputs transform."""
    transform = d4rl_policy.D4RLInputs()

    data = {
        "state": np.random.rand(29).astype(np.float32),
        "actions": np.random.rand(10, 8).astype(np.float32),
        "prompt": "antmaze-umaze-v2",
    }

    result = transform(data)

    assert "state" in result
    assert "actions" in result
    assert "prompt" in result
    assert result["state"].shape == (29,)
    assert result["actions"].shape == (10, 8)
    assert result["prompt"] == "antmaze-umaze-v2"


def test_d4rl_inputs_inference():
    """Test D4RLInputs transform without actions (inference mode)."""
    transform = d4rl_policy.D4RLInputs()

    data = {
        "state": np.random.rand(29).astype(np.float32),
        "prompt": "antmaze-umaze-v2",
    }

    result = transform(data)

    assert "state" in result
    assert "actions" not in result
    assert "prompt" in result


def test_d4rl_outputs():
    """Test D4RLOutputs transform."""
    transform = d4rl_policy.D4RLOutputs(action_dim=8)

    # Model outputs may have more dimensions than needed
    data = {"actions": np.random.rand(10, 16).astype(np.float32)}

    result = transform(data)

    assert "actions" in result
    assert result["actions"].shape == (10, 8)


def test_make_d4rl_example():
    """Test example creation."""
    example = d4rl_policy.make_d4rl_example(obs_dim=29, action_dim=8)

    assert "state" in example
    assert "prompt" in example
    assert example["state"].shape == (29,)
