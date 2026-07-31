from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def run(*command: str, cwd: Path = ROOT) -> None:
    print(f"> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    run(sys.executable, "-m", "ruff", "check", "src", "backend", "scripts", "tests")
    run(sys.executable, "-m", "mypy", "src", "backend")
    run(sys.executable, "-m", "pytest", "-q")
    run("npm.cmd", "run", "build", cwd=ROOT / "frontend")
    print("Project 12 quality gate passed.")


if __name__ == "__main__":
    main()
