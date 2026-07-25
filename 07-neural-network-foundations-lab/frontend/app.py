"""Standalone Streamlit interface for the Neural Network Foundations Lab."""

from __future__ import annotations

import html
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from neural_network_foundations.contracts import ExperimentConfig, NetworkConfig  # noqa: E402
from neural_network_foundations.datasets import DatasetBundle, get_dataset  # noqa: E402
from neural_network_foundations.evaluation import (  # noqa: E402
    binary_accuracy,
    decision_boundary,
    diagnose,
)
from neural_network_foundations.models import MLP  # noqa: E402
from neural_network_foundations.models.pytorch_reference import (  # noqa: E402
    compare_with_pytorch,
)
from neural_network_foundations.training import train  # noqa: E402


@dataclass
class LabWorkspace:
    """State persisted by Streamlit for one deterministic experiment."""

    configuration: ExperimentConfig
    dataset: DatasetBundle
    model: MLP
    initial_boundary: dict[str, Any]
    loss_history: list[float] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)
    completed_epochs: int = 0
    parity: dict[str, Any] | None = None


def build_workspace(configuration: ExperimentConfig) -> LabWorkspace:
    dataset = get_dataset(configuration.dataset, seed=configuration.seed, samples=160)
    model = MLP(configuration.network, seed=configuration.seed)
    model.forward(dataset.features)
    return LabWorkspace(
        configuration=configuration,
        dataset=dataset,
        model=model,
        initial_boundary=decision_boundary(
            model,
            dataset.features,
            resolution=configuration.grid_resolution,
        ),
    )


def train_workspace(workspace: LabWorkspace, epochs: int) -> None:
    remaining = 5_000 - workspace.completed_epochs
    if remaining <= 0:
        raise ValueError("The experiment already reached the 5,000 epoch safety limit.")
    history = train(
        workspace.model,
        workspace.dataset.features,
        workspace.dataset.targets,
        loss_name=workspace.configuration.loss,
        learning_rate=workspace.configuration.learning_rate,
        epochs=min(epochs, remaining),
    )
    workspace.loss_history.extend(history.loss)
    workspace.accuracy_history.extend(history.accuracy)
    workspace.completed_epochs += history.completed_epochs
    workspace.parity = None


def current_metrics(workspace: LabWorkspace) -> tuple[float, float]:
    predictions = workspace.model.forward(workspace.dataset.features)
    current_loss = workspace.model.calculate_loss(
        workspace.dataset.targets,
        loss_name=workspace.configuration.loss,
    )
    return current_loss, binary_accuracy(predictions, workspace.dataset.targets)


def boundary_svg(
    boundary: dict[str, Any],
    dataset: DatasetBundle,
    *,
    title: str,
) -> str:
    width, height = 640, 410
    left, top, plot_width, plot_height = 38, 54, 570, 320
    probabilities = np.asarray(boundary["probabilities"], dtype=np.float64)
    x_values = np.asarray(boundary["x"], dtype=np.float64)
    y_values = np.asarray(boundary["y"], dtype=np.float64)
    rows, columns = probabilities.shape
    cell_width, cell_height = plot_width / columns, plot_height / rows

    rectangles: list[str] = []
    for row in range(rows):
        for column in range(columns):
            probability = float(probabilities[row, column])
            red = int(29 + probability * 74)
            green = int(35 + probability * 34)
            blue = int(56 + probability * 128)
            rectangles.append(
                f'<rect x="{left + column * cell_width:.2f}" '
                f'y="{top + (rows - row - 1) * cell_height:.2f}" '
                f'width="{cell_width + 0.4:.2f}" height="{cell_height + 0.4:.2f}" '
                f'fill="rgb({red},{green},{blue})"/>'
            )

    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    points: list[str] = []
    for (x_value, y_value), target in zip(
        dataset.features,
        dataset.targets.reshape(-1),
        strict=True,
    ):
        x_position = left + ((float(x_value) - x_min) / (x_max - x_min)) * plot_width
        y_position = top + plot_height - ((float(y_value) - y_min) / (y_max - y_min)) * plot_height
        fill = "#f9ae55" if target == 0.0 else "#8dd8ff"
        points.append(
            f'<circle cx="{x_position:.2f}" cy="{y_position:.2f}" r="5.4" '
            f'fill="{fill}" stroke="#08111f" stroke-width="2"/>'
        )

    return (
        '<div class="visual-frame">'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">'
        f'<text x="32" y="30" fill="#f5f7fb" font-size="17" '
        f'font-family="Inter, sans-serif" font-weight="700">{html.escape(title)}</text>'
        f'<g opacity="0.98">{"".join(rectangles)}</g>'
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        f'rx="10" fill="none" stroke="#334155"/>{"".join(points)}'
        '<text x="38" y="397" fill="#7f8ca3" font-size="12">Class 0 · amber</text>'
        '<text x="505" y="397" fill="#7f8ca3" font-size="12">Class 1 · blue</text>'
        "</svg></div>"
    )


def dataset_svg(dataset: DatasetBundle) -> str:
    width, height = 640, 390
    left, top, plot_width, plot_height = 48, 44, 550, 300
    x_values = dataset.features[:, 0]
    y_values = dataset.features[:, 1]
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_padding = max((x_max - x_min) * 0.12, 0.15)
    y_padding = max((y_max - y_min) * 0.12, 0.15)
    x_min, x_max = x_min - x_padding, x_max + x_padding
    y_min, y_max = y_min - y_padding, y_max + y_padding
    points: list[str] = []
    for (x_value, y_value), target in zip(
        dataset.features,
        dataset.targets.reshape(-1),
        strict=True,
    ):
        x_position = left + ((float(x_value) - x_min) / (x_max - x_min)) * plot_width
        y_position = top + plot_height - ((float(y_value) - y_min) / (y_max - y_min)) * plot_height
        fill = "#f9ae55" if target == 0.0 else "#8dd8ff"
        points.append(
            f'<circle cx="{x_position:.2f}" cy="{y_position:.2f}" r="6.2" '
            f'fill="{fill}" stroke="#08111f" stroke-width="2"/>'
        )
    return (
        f'<div class="visual-frame"><svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Dataset observations">'
        '<rect x="48" y="44" width="550" height="300" rx="12" fill="#0a1020" '
        'stroke="#28364d"/>'
        '<line x1="48" y1="194" x2="598" y2="194" stroke="#263247"/>'
        '<line x1="323" y1="44" x2="323" y2="344" stroke="#263247"/>'
        f"{''.join(points)}"
        '<text x="48" y="374" fill="#7f8ca3" font-size="12">Feature 1</text>'
        '<text x="498" y="374" fill="#7f8ca3" font-size="12">amber · class 0</text>'
        '<text x="498" y="28" fill="#7f8ca3" font-size="12">blue · class 1</text>'
        "</svg></div>"
    )


def loss_svg(loss_history: list[float]) -> str:
    left, top, plot_width, plot_height = 52, 42, 540, 230
    values = np.asarray(loss_history, dtype=np.float64)
    indices = np.linspace(0, len(values) - 1, min(len(values), 240)).astype(int)
    sampled = values[indices]
    lower, upper = float(sampled.min()), float(sampled.max())
    span = max(upper - lower, 1e-12)
    points = [
        (
            left + (position / max(len(sampled) - 1, 1)) * plot_width,
            top + plot_height - ((float(value) - lower) / span) * plot_height,
        )
        for position, value in enumerate(sampled)
    ]
    path = " ".join(f"{x_value:.2f},{y_value:.2f}" for x_value, y_value in points)
    return (
        '<div class="visual-frame"><svg viewBox="0 0 640 330" role="img" '
        'aria-label="Training loss history">'
        '<defs><linearGradient id="lossGradient" x1="0" x2="1">'
        '<stop offset="0" stop-color="#a78bfa"/>'
        '<stop offset="1" stop-color="#67e8f9"/></linearGradient></defs>'
        '<rect x="52" y="42" width="540" height="230" rx="12" fill="#0a1020" '
        'stroke="#28364d"/>'
        f'<polyline points="{path}" fill="none" stroke="url(#lossGradient)" '
        'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<text x="52" y="25" fill="#aeb9ca" font-size="12">start {values[0]:.6f}</text>'
        f'<text x="442" y="25" fill="#aeb9ca" font-size="12">'
        f"current {values[-1]:.6f}</text>"
        f'<text x="52" y="305" fill="#7f8ca3" font-size="12">'
        f"{len(values):,} recorded epochs</text>"
        "</svg></div>"
    )


def network_svg(workspace: LabWorkspace) -> str:
    hidden_count = workspace.configuration.network.hidden_units
    width = 720
    height = max(350, hidden_count * 58 + 80)
    input_x, hidden_x, output_x = 90, 360, 630
    input_y = [height * 0.38, height * 0.68]
    hidden_y = np.linspace(55, height - 55, hidden_count)
    output_y = height * 0.53

    connections: list[str] = []
    for input_index, y_start in enumerate(input_y):
        for hidden_index, y_end in enumerate(hidden_y):
            weight = float(workspace.model.hidden.weights[input_index, hidden_index])
            color = "#52d6c8" if weight >= 0 else "#ff8b61"
            stroke_width = 1.2 + min(abs(weight), 2.5) * 1.4
            connections.append(
                f'<line x1="{input_x}" y1="{y_start:.2f}" x2="{hidden_x}" '
                f'y2="{y_end:.2f}" stroke="{color}" stroke-width="{stroke_width:.2f}" '
                'opacity="0.72"/>'
            )
    for hidden_index, y_start in enumerate(hidden_y):
        weight = float(workspace.model.output.weights[hidden_index, 0])
        color = "#52d6c8" if weight >= 0 else "#ff8b61"
        stroke_width = 1.2 + min(abs(weight), 2.5) * 1.4
        connections.append(
            f'<line x1="{hidden_x}" y1="{y_start:.2f}" x2="{output_x}" '
            f'y2="{output_y:.2f}" stroke="{color}" stroke-width="{stroke_width:.2f}" '
            'opacity="0.72"/>'
        )

    nodes: list[str] = []
    for index, y_value in enumerate(input_y):
        nodes.append(
            f'<circle cx="{input_x}" cy="{y_value:.2f}" r="22" fill="#101b2d" '
            'stroke="#8dd8ff" stroke-width="2"/>'
            f'<text x="{input_x}" y="{y_value + 5:.2f}" text-anchor="middle" '
            f'fill="#f5f7fb" font-size="13">x{index + 1}</text>'
        )
    for index, y_value in enumerate(hidden_y):
        nodes.append(
            f'<circle cx="{hidden_x}" cy="{y_value:.2f}" r="22" fill="#20173c" '
            'stroke="#b997ff" stroke-width="2"/>'
            f'<text x="{hidden_x}" y="{y_value + 5:.2f}" text-anchor="middle" '
            f'fill="#f5f7fb" font-size="12">h{index + 1}</text>'
        )
    nodes.append(
        f'<circle cx="{output_x}" cy="{output_y:.2f}" r="24" fill="#172d30" '
        'stroke="#52d6c8" stroke-width="2"/>'
        f'<text x="{output_x}" y="{output_y + 5:.2f}" text-anchor="middle" '
        'fill="#f5f7fb" font-size="12">ŷ</text>'
    )
    return (
        '<div class="visual-frame">'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Current neural network architecture">'
        '<text x="32" y="30" fill="#f5f7fb" font-size="17" '
        'font-family="Inter, sans-serif" font-weight="700">Current parameter graph</text>'
        f"<g>{''.join(connections)}</g><g>{''.join(nodes)}</g>"
        "</svg></div>"
    )


def apply_styles() -> None:
    stylesheet = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)


st.set_page_config(
    page_title="Neural Network Foundations Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_styles()

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">N7</div>
          <div><strong>Neural Lab</strong><span>AI Engineer · Project 07</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Experiment configuration")
    dataset_name = st.selectbox("Dataset", ("xor", "circles", "and", "or"))
    hidden_units = st.slider("Hidden neurons", 2, 8, 4)
    activation_name = st.selectbox("Hidden activation", ("tanh", "sigmoid", "relu"))
    initialization_name = st.selectbox(
        "Initialization",
        ("xavier", "he", "small_normal", "zeros"),
    )
    loss_name = st.selectbox(
        "Loss function",
        ("binary_cross_entropy", "mean_squared_error"),
    )
    learning_rate = st.slider(
        "Learning rate",
        0.0001,
        1.0,
        0.5,
        0.0001,
        format="%.4f",
    )
    requested_epochs = st.number_input(
        "Training epochs",
        min_value=1,
        max_value=5_000,
        value=1_000,
        step=100,
    )
    seed = st.number_input("Deterministic seed", min_value=0, value=7, step=1)
    grid_resolution = st.slider("Boundary resolution", 10, 70, 40)

configuration = ExperimentConfig(
    dataset=dataset_name,
    network=NetworkConfig(
        hidden_units=hidden_units,
        hidden_activation=activation_name,
        initialization=initialization_name,
    ),
    loss=loss_name,
    learning_rate=learning_rate,
    epochs=int(requested_epochs),
    seed=int(seed),
    grid_resolution=grid_resolution,
)
signature = json.dumps(configuration.to_dict(), sort_keys=True)
if "workspace" not in st.session_state or st.session_state.get("signature") != signature:
    st.session_state.workspace = build_workspace(configuration)
    st.session_state.signature = signature
    st.session_state.flash = "Experiment initialized from its deterministic configuration."

workspace: LabWorkspace = st.session_state.workspace
st.markdown(
    """
    <section class="hero">
      <div>
        <span class="eyebrow">FROM FIRST PRINCIPLES · NUMPY ENGINE</span>
        <h1>See every signal.<br><span>Understand every update.</span></h1>
        <p>Configure a bounded MLP, train it locally and inspect the values that
        make learning possible—without hiding the mathematics behind
        <code>model.fit()</code>.</p>
      </div>
      <div class="hero-badge"><span>ENGINE</span><strong>NumPy</strong>
      <small>2 → hidden → 1</small></div>
    </section>
    """,
    unsafe_allow_html=True,
)
if flash := st.session_state.pop("flash", None):
    st.success(flash, icon="✅")

current_loss, current_accuracy = current_metrics(workspace)
metric_columns = st.columns(4)
metric_columns[0].metric("Epoch", f"{workspace.completed_epochs:,}")
metric_columns[1].metric("Current loss", f"{current_loss:.6f}")
metric_columns[2].metric("Training accuracy", f"{current_accuracy:.1%}")
metric_columns[3].metric("Architecture", f"2 → {hidden_units} → 1")

action_columns = st.columns([1, 1, 1.25, 1])
if action_columns[0].button("Apply 1 update", use_container_width=True):
    train_workspace(workspace, 1)
    st.session_state.flash = "Applied one complete forward, backward and SGD update."
    st.rerun()
if action_columns[1].button("Train 100 epochs", use_container_width=True):
    train_workspace(workspace, 100)
    st.session_state.flash = "Completed 100 bounded training epochs."
    st.rerun()
if action_columns[2].button(
    f"Train {int(requested_epochs):,} epochs",
    type="primary",
    use_container_width=True,
):
    train_workspace(workspace, int(requested_epochs))
    st.session_state.flash = f"Completed {int(requested_epochs):,} epochs."
    st.rerun()
if action_columns[3].button("Reset experiment", use_container_width=True):
    st.session_state.workspace = build_workspace(configuration)
    st.session_state.flash = "Restored the deterministic initial state."
    st.rerun()

experiment_tab, trace_tab, boundary_tab, validation_tab = st.tabs(
    ("Experiment", "Forward & backward", "Decision boundary", "Validation")
)

with experiment_tab:
    left_column, right_column = st.columns([1.08, 0.92])
    with left_column:
        st.markdown("### Dataset and observed labels")
        st.markdown(dataset_svg(workspace.dataset), unsafe_allow_html=True)
        st.caption(workspace.dataset.purpose)
    with right_column:
        st.markdown("### Learning evidence")
        if workspace.loss_history:
            st.markdown(loss_svg(workspace.loss_history), unsafe_allow_html=True)
            st.caption("Training evidence on a bounded educational dataset.")
        else:
            st.info("Train at least one epoch to reveal the loss trajectory.", icon="ℹ️")
        for diagnostic in diagnose(workspace.model, workspace.loss_history):
            message = f"**{diagnostic.signal}** {diagnostic.explanation}"
            if diagnostic.severity == "success":
                st.success(message, icon="✅")
            elif diagnostic.severity == "error":
                st.error(message, icon="🚨")
            else:
                st.warning(message, icon="⚠️")

with trace_tab:
    st.markdown("### Inspect one observation through the complete computation")
    sample_index = st.selectbox(
        "Observation",
        range(workspace.dataset.features.shape[0]),
        format_func=lambda index: (
            f"#{index + 1} · x={workspace.dataset.features[index].tolist()} "
            f"· target={int(workspace.dataset.targets[index, 0])}"
        ),
    )
    workspace.model.forward(workspace.dataset.features)
    workspace.model.backward(workspace.dataset.targets, loss_name=configuration.loss)
    trace = workspace.model.trace_sample(
        dataset=workspace.dataset.name,
        features=workspace.dataset.features,
        targets=workspace.dataset.targets,
        sample_index=sample_index,
        loss_name=configuration.loss,
        configuration=configuration.to_dict(),
    )
    trace_columns = st.columns([1.05, 0.95])
    with trace_columns[0]:
        st.markdown(network_svg(workspace), unsafe_allow_html=True)
    with trace_columns[1]:
        st.markdown(
            f"""
            <div class="trace-summary"><span>SELECTED OBSERVATION</span>
            <strong>Prediction {trace.prediction:.6f}</strong>
            <p>Target {trace.target:.0f} · sample loss {trace.loss:.6f}</p></div>
            """,
            unsafe_allow_html=True,
        )
        node_options = {f"{node.layer_id} · {node.neuron_id}": node for node in trace.nodes}
        selected_node = node_options[st.selectbox("Neuron detail", tuple(node_options))]
        st.json(
            {
                "inputs": selected_node.inputs,
                "weights": selected_node.weights,
                "bias": selected_node.bias,
                "z = xW + b": selected_node.z,
                "activation": selected_node.activation_name,
                "activation_value": selected_node.activation_value,
                "upstream_gradient": selected_node.upstream_gradient,
                "local_gradient": selected_node.local_gradient,
                "parameter_gradients": selected_node.parameter_gradients,
            },
            expanded=True,
        )

with boundary_tab:
    st.markdown("### Compare initialization with the learned representation")
    current_boundary = decision_boundary(
        workspace.model,
        workspace.dataset.features,
        resolution=configuration.grid_resolution,
    )
    before_column, after_column = st.columns(2)
    with before_column:
        st.markdown(
            boundary_svg(
                workspace.initial_boundary,
                workspace.dataset,
                title="Before training",
            ),
            unsafe_allow_html=True,
        )
    with after_column:
        st.markdown(
            boundary_svg(
                current_boundary,
                workspace.dataset,
                title=f"After {workspace.completed_epochs:,} epochs",
            ),
            unsafe_allow_html=True,
        )
    st.caption("Background color is model probability; points are training observations.")

with validation_tab:
    st.markdown("### Independent mathematical checks")
    validation_columns = st.columns([0.82, 1.18])
    with validation_columns[0]:
        if st.button("Compare NumPy with PyTorch", type="primary", use_container_width=True):
            with st.spinner("Running an independent float64 parity check..."):
                workspace.parity = compare_with_pytorch(
                    workspace.model,
                    workspace.dataset.features,
                    workspace.dataset.targets,
                    loss_name=configuration.loss,
                    learning_rate=configuration.learning_rate,
                ).to_dict()
        if workspace.parity is None:
            st.info(
                "Run parity to compare forward, loss, gradients and one update.",
                icon="ℹ️",
            )
        elif workspace.parity["passed"]:
            st.success("NumPy and PyTorch agree within tolerance.", icon="✅")
        else:
            st.error("Parity exceeded the approved tolerance.", icon="🚨")
    with validation_columns[1]:
        if workspace.parity is not None:
            parity_rows = [
                ("Forward", workspace.parity["forward_max_absolute_error"]),
                ("Loss", workspace.parity["loss_absolute_error"]),
                ("Input gradient", workspace.parity["input_gradient_max_absolute_error"]),
                *[
                    (f"Gradient · {name}", value)
                    for name, value in workspace.parity["gradient_max_absolute_error"].items()
                ],
            ]
            rows_html = "".join(
                f"<tr><td>{html.escape(label)}</td><td>{value:.3e}</td></tr>"
                for label, value in parity_rows
            )
            st.markdown(
                '<table class="parity-table"><thead><tr><th>Check</th>'
                f"<th>Maximum absolute error</th></tr></thead><tbody>{rows_html}"
                "</tbody></table>",
                unsafe_allow_html=True,
            )
            st.caption(f"Approved tolerance: {workspace.parity['tolerance']:.1e}")

st.markdown(
    """
    <footer><span>Project 07 · Neural Network Foundations Lab</span>
    <span>NumPy first · deterministic · CPU bounded</span></footer>
    """,
    unsafe_allow_html=True,
)
