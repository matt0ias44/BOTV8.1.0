#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convenience launcher for the live trading stack (ingestion + inference + trader)."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

DEFAULT_COMPONENTS: Dict[str, List[str]] = {
    "rss": ["node", "rss_to_csv.js"],
    "price": [PYTHON, "live/price_pipe_writer.py"],
    "bridge": [PYTHON, "bridge_inference.py"],
    "trader": [PYTHON, "live_trader.py"],
}

STREAMLIT_CMD = [PYTHON, "-m", "streamlit", "run", "app.py"]


def build_commands(include_streamlit: bool) -> Dict[str, List[str]]:
    commands = dict(DEFAULT_COMPONENTS)
    if include_streamlit:
        commands["dashboard"] = STREAMLIT_CMD
    return commands


def start_process(name: str, cmd: List[str]) -> subprocess.Popen:
    print(f"[launcher] starting {name}: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def stream_output(name: str, proc: subprocess.Popen) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(f"[{name}] {line}")
    proc.stdout.close()


def run_stack(include_streamlit: bool) -> int:
    commands = build_commands(include_streamlit)
    procs: Dict[str, subprocess.Popen] = {}
    threads: List[threading.Thread] = []

    try:
        for name, cmd in commands.items():
            proc = start_process(name, cmd)
            procs[name] = proc
            t = threading.Thread(target=stream_output, args=(name, proc), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(0.5)

        print("[launcher] live stack running. Press Ctrl+C to stop.")
        while True:
            exited = [n for n, p in procs.items() if p.poll() is not None]
            if exited:
                for n in exited:
                    print(f"[launcher] process '{n}' exited with code {procs[n].returncode}")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[launcher] received interrupt, shutting down...")
    finally:
        for name, proc in procs.items():
            if proc.poll() is None:
                try:
                    if os.name == "nt":
                        proc.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        proc.send_signal(signal.SIGINT)
                    time.sleep(0.5)
                except Exception:
                    pass
        for proc in procs.values():
            if proc.poll() is None:
                proc.terminate()
        time.sleep(1)
        for proc in procs.values():
            if proc.poll() is None:
                proc.kill()

    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live trading stack.")
    parser.add_argument(
        "--with-dashboard",
        action="store_true",
        help="Also launch the Streamlit dashboard (streamlit run app.py).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return run_stack(include_streamlit=args.with_dashboard)


if __name__ == "__main__":
    sys.exit(main())
