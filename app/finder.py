import os
from pathlib import Path

from .manage import load_config


def find_model(model_name: str) -> Path | None:
    """
    Search for a given model name in ~/.lmstudio/models, ~/.cache/huggingface/hub,
    and any custom paths added via manage-model-sources.
    Returns the Path to the directory containing config.json if found.
    """
    home = Path.home()

    # Default places to search
    search_paths = [
        home / ".lmstudio" / "models",
        home / ".cache" / "huggingface" / "hub",
    ]

    # Add custom paths from config
    config = load_config()
    custom_paths = config.get("custom_paths", [])
    for p in custom_paths:
        search_paths.append(Path(p))

    # We are looking for directories that:
    # 1. contain the `model_name` string in their path
    # 2. contain `config.json`

    for base_path in search_paths:
        if not base_path.exists():
            continue

        # Walking through the directory
        for root, _dirs, files in os.walk(base_path):
            if "config.json" in files and model_name.lower() in str(Path(root)).lower():
                # We might want to verify it's a valid mlx-lm layout
                # Usually just config.json is enough, but we could check tokenizer too.
                return Path(root)

    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        res = find_model(sys.argv[1])
        if res:
            print(f"Found model at: {res}")
        else:
            print(f"Model {sys.argv[1]} not found.")
    else:
        print("Usage: python -m app.finder <model_name>")
