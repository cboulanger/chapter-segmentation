"""Starts a local static server for the repo root and opens the
ground-truth review app (evaluation/app/) in the browser, pointed at one
corpus. macOS only (uses the `open` and `lsof` commands) -- see
evaluation/app/README.md and
docs/superpowers/specs/2026-08-11-ground-truth-review-app-design.md.

    uv run review open-access
    uv run review open-access --index 5
    uv run review open-access --port 8080
    uv run review-stop
    uv run review-stop --port 8080
"""

import argparse
import functools
import http.server
import os
import signal
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORT = 8743


def _pids_on_port(port):
    """PIDs of any process listening on `port` (macOS `lsof`), or []."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return []
    return [int(pid) for pid in result.stdout.split()]


def main():
    parser = argparse.ArgumentParser(description="Serve and open the ground-truth review app for one corpus.")
    parser.add_argument("corpus", help="corpus directory name under evaluation/corpus/, e.g. open-access")
    parser.add_argument("--index", type=int, default=0, help="starting book index (default: 0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"local server port (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    corpus_dir = REPO_ROOT / "evaluation" / "corpus" / args.corpus
    if not corpus_dir.is_dir():
        sys.exit(f"No such corpus directory: {corpus_dir}")

    url = (
        f"http://localhost:{args.port}/evaluation/app/index.html"
        f"?corpus={args.corpus}&index={args.index}&repoRoot={quote(str(REPO_ROOT), safe='')}"
    )

    if _pids_on_port(args.port):
        print(f"A server is already listening on port {args.port}; reusing it.")
        print(f"Opening {url}")
        subprocess.run(["open", url], check=True)
        return

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(REPO_ROOT))
    try:
        httpd = http.server.HTTPServer(("localhost", args.port), handler)
    except OSError as err:
        sys.exit(f"Could not bind port {args.port}: {err}. Run `uv run review-stop --port {args.port}` and retry.")

    with httpd:
        print(f"Serving {REPO_ROOT} at http://localhost:{args.port}/")
        print(f"Opening {url}")
        subprocess.run(["open", url], check=True)
        print("Press Ctrl+C to stop, or run `uv run review-stop` from elsewhere.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def stop():
    parser = argparse.ArgumentParser(description="Stop whatever review server is listening on a port.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to stop (default: {DEFAULT_PORT})")
    args = parser.parse_args()

    pids = _pids_on_port(args.port)
    if not pids:
        print(f"No review server is running on port {args.port}.")
        return
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    print(f"Stopped review server on port {args.port} (pid {', '.join(map(str, pids))}).")


if __name__ == "__main__":
    main()
