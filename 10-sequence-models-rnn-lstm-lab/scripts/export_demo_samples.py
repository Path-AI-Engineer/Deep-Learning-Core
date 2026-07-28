from __future__ import annotations

import json
import sys

from _common import PROJECT_ROOT, SRC_ROOT, write_json

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sequence_models.data import build_demo_records  # noqa: E402


def main() -> None:
    records = build_demo_records(2)
    payload = [
        {
            "sample_id": record.sample_id,
            "label": record.label,
            "activity": record.activity,
            "subject_id": record.subject_id,
            "shape": list(record.values.shape),
            "values": record.values.tolist(),
            "evidence_status": "deterministic_educational_fixture",
        }
        for record in records
    ]
    destination = PROJECT_ROOT / "reports" / "runs" / "demo_samples.json"
    write_json(destination, payload)
    print(json.dumps({"samples": len(payload), "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()

