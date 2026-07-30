import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Start Server with Specific Context Size Profiles")
    parser.add_argument(
        "model",
        type=str,
        help="The name of the model to load from local cache."
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[4096, 8192, 16384, 32768, 65536],
        default=8192,
        help="Context window size profile to use (default: 8192)."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the HTTP server"
    )

    args = parser.parse_args()

    # Pre-fill step size should be ~20-25% less than context size
    # So we multiply by 0.75 for a safe margin
    prefill_step_size = int(args.size * 0.75)
    max_tokens = args.size

    print("Starting server with profile:")
    print(f" - Model: {args.model}")
    print(f" - Max Context (tokens): {max_tokens}")
    print(f" - Prefill Step Size: {prefill_step_size}")

    # Build the command to run the main server.py
    cmd = [
        "start-inference-server",
        args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--max-tokens", str(max_tokens),
        "--prefill-step-size", str(prefill_step_size)
    ]

    # We use subprocess to call the other entry point
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("Shutting down context server launcher...")
    except subprocess.CalledProcessError as e:
        print(f"Server exited with error code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        # Fallback if the script isn't on the path yet (e.g. running from source)
        cmd[0] = sys.executable
        cmd.insert(1, "-m")
        cmd.insert(2, "app.server")
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("Shutting down context server launcher...")

if __name__ == "__main__":
    main()
