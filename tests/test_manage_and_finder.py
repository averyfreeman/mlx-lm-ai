from unittest.mock import patch

import pytest

from app.finder import find_model
from app.manage import load_config, save_config


@pytest.fixture
def mock_home(tmp_path):
    with patch("pathlib.Path.home", return_value=tmp_path):
        yield tmp_path

def test_load_save_config(mock_home):
    # Ensure default is empty list
    config = load_config()
    assert config == {"custom_paths": []}

    # Save a custom path
    config["custom_paths"].append("/tmp/fake_models")
    save_config(config)

    # Load and verify
    loaded = load_config()
    assert loaded == {"custom_paths": ["/tmp/fake_models"]}

def test_finder_resolves_custom_path(mock_home):
    # Create a mock config with a custom path
    custom_dir = mock_home / "my_custom_models"
    custom_dir.mkdir()

    config = {"custom_paths": [str(custom_dir)]}
    save_config(config)

    # Create a fake model inside the custom path
    model_dir = custom_dir / "Qwen3.5-9B"
    model_dir.mkdir()
    (model_dir / "config.json").touch()

    # Run the finder
    result = find_model("Qwen3.5-9B")
    assert result == model_dir

def test_finder_resolves_default_paths(mock_home):
    # Create a fake model inside the default lmstudio path
    lmstudio_dir = mock_home / ".lmstudio" / "models" / "lmstudio-community" / "LFM2-24B-A2B-MLX-4bit"
    lmstudio_dir.mkdir(parents=True)
    (lmstudio_dir / "config.json").touch()

    # Run the finder
    result = find_model("LFM2-24B")
    assert result == lmstudio_dir
