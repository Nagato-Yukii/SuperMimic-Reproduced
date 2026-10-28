"""Plain YAML presets use the same Tyro validation and CLI override behavior."""

from pathlib import Path

import pytest
import tyro

from supermimic.config import TrainConfig
from supermimic.config_yaml import expand_config_args


def test_velocity_preset_and_cli_override():
    preset = Path(__file__).resolve().parents[1] / "configs/motion_velocity_5070ti.yaml"
    cfg = tyro.cli(
        TrainConfig,
        args=expand_config_args(["--config", str(preset), "--schedule.total-iterations", "20"]),
    )
    assert cfg.task == "motion"
    assert cfg.env.num_envs == 4096
    assert cfg.policy.velocity_encoder
    assert cfg.policy.latent_dim == 32
    assert cfg.schedule.total_iterations == 20


def test_invalid_yaml_shape(tmp_path):
    preset = tmp_path / "bad.yaml"
    preset.write_text("- task: motion\n")
    with pytest.raises(TypeError, match="must be a mapping"):
        expand_config_args(["--config", str(preset)])
