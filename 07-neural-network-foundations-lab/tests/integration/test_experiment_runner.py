from pathlib import Path
from uuid import uuid4

from neural_network_foundations.contracts import ExperimentConfig
from neural_network_foundations.experiments import run_experiment
from neural_network_foundations.serialization import read_json


def test_experiment_runner_persists_reconstructable_evidence() -> None:
    output = Path(".runtime/tests") / f"runner-{uuid4().hex}"
    result = run_experiment(
        ExperimentConfig(epochs=20, grid_resolution=10),
        artifact_root=output,
        run_id="controlled-run",
        include_parity=False,
    )
    assert result.status == "completed"
    assert result.gradient_check_passed
    assert result.parity_passed is None
    assert read_json(output / "controlled-run" / "config.json")["seed"] == 7
    assert (output / "controlled-run" / "checkpoint.npz").exists()
    assert (output / "controlled-run" / "summary.md").exists()
