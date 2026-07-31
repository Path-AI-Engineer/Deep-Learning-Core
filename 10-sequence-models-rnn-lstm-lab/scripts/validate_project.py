from __future__ import annotations

import json
import sys

from _common import PROJECT_ROOT, SRC_ROOT

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app  # noqa: E402
from sequence_models.artifacts import load_bundle  # noqa: E402
from sequence_models.cells import cell_trace  # noqa: E402


def main() -> None:
    required = (
        "README.md",
        "infra/docker/production.Dockerfile",
        "frontend/package-lock.json",
        "docs/model-card.md",
        "docs/data-contract.md",
        "artifacts/comparisons/v1.0.0/model_comparison.json",
    )
    missing = [name for name in required if not (PROJECT_ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")
    bundles = {}
    for model_name in ("rnn", "lstm", "gru"):
        bundle = load_bundle(PROJECT_ROOT / "artifacts" / "models" / model_name / "v1.0.0")
        bundles[model_name] = bundle.manifest["state_sha256"]
    route_paths = {
        route.path
        for route in app.routes
        if isinstance(getattr(route, "path", None), str)
        and str(route.path).startswith("/api/v1")
    }
    if len(route_paths) != 14:
        raise SystemExit(f"Expected 14 API paths, found {len(route_paths)}")
    parity = {
        name: cell_trace(name).max_abs_difference
        for name in ("rnn", "lstm", "gru")
    }
    if any(value > 1e-9 for value in parity.values()):
        raise SystemExit(f"Cell parity failed: {parity}")
    broken_markers = ("Ã", "â€", "ðŸ", "\ufffd")
    broken_files: list[str] = []
    for path in (
        list(PROJECT_ROOT.glob("*.md"))
        + list((PROJECT_ROOT / "docs").glob("*.md"))
        + list((PROJECT_ROOT / "labs").glob("*/README.md"))
    ):
        if any(marker in path.read_text(encoding="utf-8") for marker in broken_markers):
            broken_files.append(str(path.relative_to(PROJECT_ROOT)))
    if broken_files:
        raise SystemExit(f"Broken text encoding found in: {broken_files}")
    print(
        json.dumps(
            {
                "status": "passed",
                "api_paths": len(route_paths),
                "bundle_hashes": bundles,
                "cell_parity": parity,
                "encoding": "clean",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
