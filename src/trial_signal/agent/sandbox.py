"""Run agent-written code in a locked-down container.

Build once:  docker build -t trial-signal-sandbox sandbox/
Docs:        https://docs.docker.com/reference/cli/docker/container/run/
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

IMAGE = "trial-signal-sandbox"

# Applies the agent's extract() to every sample and prints one JSON array. Errors are caught
# per record so one bad record doesn't hide the rest.
_APPLY = """
import json, pathlib, traceback
from extract import extract
out = []
for p in sorted(pathlib.Path('/samples').glob('*.json')):
    study = json.loads(p.read_text())
    try:
        out.append({'file': p.name, 'output': extract(study)})
    except Exception:
        out.append({'file': p.name, 'error': traceback.format_exc(limit=2)[-500:]})
print(json.dumps(out, default=str))
"""


@dataclass
class SandboxResult:
    passed: bool
    output: str


def _docker(workspace: Path, samples: Path, *argv: str) -> list[str]:
    return [
        "docker", "run", "--rm",
        "--network", "none",              # no exfiltration, no surprise API calls
        "--read-only",                    # container filesystem is immutable...
        "--tmpfs", "/tmp:rw,size=64m",    # ...except a small scratch area
        "--memory", "512m", "--cpus", "1",
        "--pids-limit", "128",
        "--user", "65534:65534",          # nobody
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", f"{workspace.resolve()}:/work:ro",
        "-v", f"{samples.resolve()}:/samples:ro",
        "-w", "/work",
        IMAGE,
        *argv,
    ]


def run_pytest(workspace: Path, samples: Path, timeout_s: int = 120) -> SandboxResult:
    cmd = _docker(workspace, samples, "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_extract.py")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return SandboxResult(False, f"timed out after {timeout_s}s")
    out = (proc.stdout + proc.stderr)[-4000:]
    return SandboxResult(proc.returncode == 0, out)


def apply_extract(workspace: Path, samples: Path, timeout_s: int = 300) -> list[dict]:
    """[{'file', 'output'} | {'file', 'error'}, ...] for every sample. Raises if the run itself fails."""
    cmd = _docker(workspace, samples, "python", "-c", _APPLY)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"extract run failed: {(proc.stdout + proc.stderr)[-2000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])
