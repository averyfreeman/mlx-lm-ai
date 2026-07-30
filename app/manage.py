import argparse
import json
import sys
from pathlib import Path


def get_config_path() -> Path:
    return Path.home() / ".mlx_server_config.json"


def load_config() -> dict:
    config_path = get_config_path()
    if not config_path.exists():
        return {"custom_paths": []}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return {"custom_paths": []}
    except json.JSONDecodeError as e:
        print(f"JSON decode error in config: {e}", file=sys.stderr)
        return {"custom_paths": []}


def save_config(config: dict) -> None:
    config_path = get_config_path()
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
    except OSError as e:
        print(f"Error saving config: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Manage Model Sources",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command
    _list_parser = subparsers.add_parser("list", help="List custom model directories")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a custom model directory")
    add_parser.add_argument("path", type=str, help="The directory path to add")

    # Remove command
    remove_parser = subparsers.add_parser(
        "remove", help="Remove a custom model directory"
    )
    remove_parser.add_argument("path", type=str, help="The directory path to remove")

    args = parser.parse_args()

    config = load_config()
    custom_paths = config.get("custom_paths", [])

    if args.command == "list":
        if not custom_paths:
            print("No custom model directories configured.")
        else:
            print("Custom model directories:")
            for p in custom_paths:
                print(f"  - {p}")

    elif args.command == "add":
        path_obj = Path(args.path).resolve()
        path_str = str(path_obj)
        if not path_obj.exists() or not path_obj.is_dir():
            print(
                f"Warning: The path '{path_str}' does not exist or is not a directory.",
                file=sys.stderr,
            )

        if path_str not in custom_paths:
            custom_paths.append(path_str)
            config["custom_paths"] = custom_paths
            save_config(config)
            print(f"Added custom path: {path_str}")
        else:
            print(f"Path '{path_str}' is already configured.")

    elif args.command == "remove":
        path_obj = Path(args.path).resolve()
        path_str = str(path_obj)
        if path_str in custom_paths:
            custom_paths.remove(path_str)
            config["custom_paths"] = custom_paths
            save_config(config)
            print(f"Removed custom path: {path_str}")
        else:
            # Also try to match directly by the user input just in case
            if args.path in custom_paths:
                custom_paths.remove(args.path)
                config["custom_paths"] = custom_paths
                save_config(config)
                print(f"Removed custom path: {args.path}")
            else:
                print(f"Path not found in configuration: {args.path}")


if __name__ == "__main__":
    main()
