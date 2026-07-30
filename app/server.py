import argparse
import sys

import uvicorn
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(description="MLX Dynamic Model Server")
    parser.add_argument(
        "model", type=str, help="The name of the model to load from local cache."
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host for the HTTP server"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for the HTTP server"
    )
    parser.add_argument(
        "--fix-mistral-regex",
        action="store_true",
        help="Fix mistral regex tokenizer issue",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens (used for context wrappers)",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=None,
        help="Prefill step size (used for context wrappers)",
    )
    return parser.parse_args()


def main():
    args = get_args()
    model_name = args.model

    from .finder import find_model

    print(f"Searching for model matching '{model_name}'...")
    model_path = find_model(model_name)

    if not model_path:
        print(
            f"Error: Could not find model '{model_name}' in ~/.lmstudio/models or ~/.cache/huggingface/hub.",
            file=sys.stderr,
        )
        print(
            "Please ensure it has been downloaded and contains a config.json file.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found model at: {model_path}")
    print("Loading model...")

    # We defer mlx_lm imports to allow running/testing scripts in environments
    # without MLX hardware support directly. The user runs on an M2 mac so it will succeed.
    from mlx_lm import load
    from mlx_lm.server import APIHandler, LRUPromptCache, ResponseGenerator

    try:
        custom_tokenizer = AutoTokenizer.from_pretrained(
            model_path, fix_mistral_regex=args.fix_mistral_regex
        )
        tokenizer_config = {"tokenizer": custom_tokenizer}
    except OSError as e:
        print(f"Warning: Failed to load custom tokenizer using transformers: {e}")
        tokenizer_config = {}

    model, tokenizer = load(str(model_path), tokenizer_config=tokenizer_config)

    class FakeModelProvider:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.model_name = str(model_path)

            class Args:
                def __init__(self):
                    self.model = str(model_path)
                    self.chat_template = ""
                    self.use_default_chat_template = False
                    self.max_tokens = args.max_tokens or 2048
                    self.temp = 0.7
                    self.top_p = 1.0
                    self.min_p = 0.0
                    self.prompt_cache_size = 10
                    self.prefill_step_size = args.prefill_step_size or 2048

            self.cli_args = Args()

    provider = FakeModelProvider(model, tokenizer)
    prompt_cache = LRUPromptCache(provider.cli_args.prompt_cache_size)
    response_generator = ResponseGenerator(provider, prompt_cache)

    try:
        handler = APIHandler(response_generator)
        app = handler.app
    except TypeError:
        handler = APIHandler(model, tokenizer)
        app = handler.app

    print(f"Starting server on http://{args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
