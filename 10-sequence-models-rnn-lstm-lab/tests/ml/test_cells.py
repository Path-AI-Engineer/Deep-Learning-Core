import numpy as np
import pytest

from sequence_models.cells import cell_trace, rnn_step


@pytest.mark.parametrize("cell_type", ["rnn", "lstm", "gru"])
def test_educational_cells_match_pytorch(cell_type: str) -> None:
    trace = cell_trace(cell_type)
    assert trace.max_abs_difference <= trace.parity_tolerance
    assert len(trace.timesteps) == 3


def test_lstm_gates_stay_in_expected_ranges() -> None:
    trace = cell_trace("lstm")
    for step in trace.timesteps:
        gates = step["gates"]
        for name in ("input", "forget", "output"):
            assert all(0.0 <= value <= 1.0 for value in gates[name])
        assert all(-1.0 <= value <= 1.0 for value in gates["candidate"])


def test_gru_has_no_separate_cell_state() -> None:
    trace = cell_trace("gru")
    assert all("cell" not in step for step in trace.timesteps)


def test_rnn_rejects_incompatible_weight_shapes() -> None:
    with pytest.raises(ValueError, match="weight_ih"):
        rnn_step(
            np.ones(2),
            np.ones(3),
            np.ones((2, 2)),
            np.ones((3, 3)),
            np.ones(3),
            np.ones(3),
        )
