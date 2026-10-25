"""Run agent-written code in a locked-down container.

Build once:  docker build -t trial-signal-sandbox sandbox/
Docs:        https://docs.docker.com/reference/cli/docker/container/run/
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

IMAGE = "trial-signal-sandbox"


@dataclass
class SandboxResult:
    passed: bool
    output: str


def run_pytest(workspace: Path, samples: Path, timeout_s: int = 120) -> SandboxResult:
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",              # no exfiltration, no surprise API calls
        "--read-only",                    # container filesystem is immutable...
        "--tmpfs", "/tmp:rw,size=64m",    # ...except a small scratch area
        "--memory", "512m", "--cpus", "1",
        "--pids-limit", "128",
        "--user", "65534:65534",          # nobody
        "-v", f"{workspace.resolve()}:/work:ro",
        "-v", f"{samples.resolve()}:/samples:ro",
        "-w", "/work",
        IMAGE,
        "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_extract.py",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return SandboxResult(False, f"timed out after {timeout_s}s")
    out = (proc.stdout + proc.stderr)[-4000:]
    return SandboxResult(proc.returncode == 0, out)
