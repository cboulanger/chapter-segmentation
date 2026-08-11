"""Starts a local static server for the repo root and opens the
ground-truth review app (evaluation/app/) in the browser, pointed at one
corpus. macOS only (uses the `open` command) -- see evaluation/app/README.md
and docs/superpowers/specs/2026-08-11-ground-truth-review-app-design.md.

    uv run review open-access
    uv run review open-access --index 5
    uv run review open-access --port 8080
"""

import argparse
import functools
import http.server
import socketserver
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 8743


def main():
    parser = argparse.ArgumentParser(description="Serve and open the ground-truth review app for one corpus.")
    parser.add_argument("corpus", help="corpus directory name under evaluation/corpus/, e.g. open-access")
    parser.add_argument("--index", type=int, default=0, help="starting book index (default: 0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"local server port (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    corpus_dir = REPO_ROOT / "evaluation" / "corpus" / args.corpus
    if not corpus_dir.is_dir():
        sys.exit(f"No such corpus directory: {corpus_dir}")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(REPO_ROOT))
    with socketserver.TCPServer(("localhost", args.port), handler) as httpd:
        url = f"http://localhost:{args.port}/evaluation/app/index.html?corpus={args.corpus}&index={args.index}"
        print(f"Serving {REPO_ROOT} at http://localhost:{args.port}/")
        print(f"Opening {url}")
        subprocess.run(["open", url], check=True)
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
