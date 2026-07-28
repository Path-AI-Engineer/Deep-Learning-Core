from __future__ import annotations

import json
import sys

from _common import PROJECT_ROOT, SRC_ROOT, write_json

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sequence_models.cells import cell_trace  # noqa: E402
from sequence_models.experiments import gradient_flow_experiment  # noqa: E402


def main() -> None:
    output = PROJECT_ROOT / "reports" / "runs" / "cell-labs"
    traces = {name: cell_trace(name).to_dict() for name in ("rnn", "lstm", "gru")}
    gradient = gradient_flow_experiment()
    write_json(output / "cell_parity.json", traces)
    write_json(output / "gradient_flow.json", gradient)
    payload = {
        "cell_parity": {
            name: trace["max_abs_difference"] for name, trace in traces.items()
        },
        "gradient_flow_finite": gradient["finite"],
        "output": str(output),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

