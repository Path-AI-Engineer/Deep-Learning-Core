"""Metrics and mathematical verification."""

from neural_network_foundations.evaluation.boundary import decision_boundary
from neural_network_foundations.evaluation.diagnostics import Diagnostic, diagnose
from neural_network_foundations.evaluation.gradient_check import (
    GradientCheckReport,
    check_model_gradients,
)
from neural_network_foundations.evaluation.metrics import binary_accuracy

__all__ = [
    "Diagnostic",
    "GradientCheckReport",
    "binary_accuracy",
    "check_model_gradients",
    "decision_boundary",
    "diagnose",
]
