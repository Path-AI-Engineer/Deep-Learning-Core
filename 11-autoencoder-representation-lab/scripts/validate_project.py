from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _common import PROJECT_ROOT


def run(label: str, command: list[str], cwd: Path = PROJECT_ROOT) -> None:
    print(f"\n[{label}] {' '.join(command)}")
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise SystemExit(f"{label} failed with exit code {completed.returncode}")


def main() -> None:
    run("artifact", [sys.executable, "scripts/build_bundle.py", "--model", "conv-ae"])
    run("ruff", [sys.executable, "-m", "ruff", "check", ".", "--no-cache"])
    run("mypy", [sys.executable, "-m", "mypy", "--no-incremental"])
    run("pytest", [sys.executable, "-m", "pytest", "-q"])
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    run("frontend", [npm, "run", "build"], PROJECT_ROOT / "frontend")
    print("\nProject 11 quality gate passed.")


if __name__ == "__main__":
    main()
