"""Smoke tests for the standalone Project 07 interface."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[2] / "frontend" / "app.py"


def test_interface_renders_initial_experiment() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert [metric.value for metric in app.metric[:3]] == ["0", "0.706439", "25.0%"]
    assert [button.label for button in app.button[:4]] == [
        "Apply 1 update",
        "Train 100 epochs",
        "Train 1,000 epochs",
        "Reset experiment",
    ]


def test_interface_applies_real_training_update() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    app.button[0].click().run()

    assert not app.exception
    assert app.metric[0].value == "1"
    assert float(app.metric[1].value) < 0.706439
    assert any("forward, backward and SGD update" in item.value for item in app.success)
