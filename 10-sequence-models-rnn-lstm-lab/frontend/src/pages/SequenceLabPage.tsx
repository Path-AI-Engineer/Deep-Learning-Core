import { useEffect, useState } from "react";
import { api } from "../api/client";
import { MiniLineChart } from "../components/Charts";
import { MemoryIcon } from "../components/Icons";
import { ErrorPanel, LoadingPanel, PageHeader, StatusBanner } from "../components/Primitives";
import type { CellTrace, GradientFlow, ModelId } from "../types/contracts";

export function SequenceLabPage() {
  const [cellType, setCellType] = useState<ModelId>("lstm");
  const [trace, setTrace] = useState<CellTrace | null>(null);
  const [gradient, setGradient] = useState<GradientFlow | null>(null);
  const [timestep, setTimestep] = useState(0);
  const [clipping, setClipping] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let ignore = false;
    setTrace(null);
    setTimestep(0);
    api.cellTrace(cellType).then((result) => !ignore && setTrace(result)).catch((reason: Error) => !ignore && setError(reason.message));
    return () => { ignore = true; };
  }, [cellType]);

  useEffect(() => {
    let ignore = false;
    api.gradientFlow().then((result) => !ignore && setGradient(result)).catch((reason: Error) => !ignore && setError(reason.message));
    return () => { ignore = true; };
  }, []);

  const step = trace?.timesteps[timestep];
  return (
    <div className="page">
      <PageHeader eyebrow="States · gates · gradients" title="Step inside recurrent memory." description="Advance one timestep at a time, compare the educational equations with PyTorch and observe how gradient paths change with sequence length." />
      {error && <ErrorPanel message={error} />}
      <div className="segmented" role="group" aria-label="Cell type">
        {(["rnn", "lstm", "gru"] as ModelId[]).map((item) => <button key={item} className={cellType === item ? "active" : ""} onClick={() => setCellType(item)}>{item.toUpperCase()}</button>)}
      </div>
      {!trace ? <LoadingPanel label="Calculating cell parity" /> : (
        <section className="cell-lab">
          <div className="cell-stage">
            <div className="cell-heading"><span className="icon-box"><MemoryIcon /></span><div><p className="eyebrow">Balanced memory example</p><h2>{cellType.toUpperCase()}Cell · timestep {timestep + 1}</h2></div><span className={trace.max_abs_difference <= trace.parity_tolerance ? "parity-pass" : "parity-fail"}>{trace.max_abs_difference <= trace.parity_tolerance ? "Parity passed" : "Parity failed"}</span></div>
            <div className="state-flow">
              <div><span>xₜ</span><strong>[{step?.input.map((value) => value.toFixed(2)).join(", ")}]</strong><small>Current input</small></div>
              <i>+</i>
              <div><span>hₜ₋₁</span><strong>[{step?.previous_hidden.map((value) => value.toFixed(2)).join(", ")}]</strong><small>Previous memory</small></div>
              <i>→</i>
              <div className="state-current"><span>hₜ</span><strong>[{step?.hidden.map((value) => value.toFixed(2)).join(", ")}]</strong><small>Updated state</small></div>
            </div>
            {step?.gates && <div className="gate-grid">{Object.entries(step.gates).map(([name, values]) => <article key={name}><div><span>{name} gate</span><strong>{(values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(3)}</strong></div><div className="gate-track"><i style={{ width: `${Math.max(0, Math.min(1, values.reduce((sum, value) => sum + value, 0) / values.length)) * 100}%` }} /></div><small>Mean activation · bounded 0–1 where sigmoid applies</small></article>)}</div>}
            {step?.cell && <div className="cell-state-callout"><span>Cell state cₜ</span><strong>[{step.cell.map((value) => value.toFixed(3)).join(", ")}]</strong><small>Additive memory path can contain negative values.</small></div>}
            <label className="timestep-control">Timestep <strong>{timestep + 1} / {trace.timesteps.length}</strong><input type="range" min="0" max={trace.timesteps.length - 1} value={timestep} onChange={(event) => setTimestep(Number(event.target.value))} /></label>
          </div>
          <aside className="parity-panel">
            <p className="eyebrow">Numerical proof</p><h2>Educational vs PyTorch</h2>
            <div><span>Educational</span><code>{trace.educational_output.map((value) => value.toFixed(6)).join("  ")}</code></div>
            <div><span>PyTorch cell</span><code>{trace.pytorch_output.map((value) => value.toFixed(6)).join("  ")}</code></div>
            <dl><dt>Max absolute difference</dt><dd>{trace.max_abs_difference.toExponential(2)}</dd><dt>Tolerance</dt><dd>{trace.parity_tolerance.toExponential(1)}</dd></dl>
            <StatusBanner kind="success">The educational cell follows the installed PyTorch gate order and equation.</StatusBanner>
          </aside>
        </section>
      )}
      <section className="panel gradient-panel">
        <div className="panel-heading"><div><p className="eyebrow">Backpropagation through time</p><h2>Gradient flow by sequence length</h2><p>Repeated Jacobian products can shrink or amplify the signal reaching early timesteps.</p></div><label className="switch"><input type="checkbox" checked={clipping} onChange={(event) => setClipping(event.target.checked)} /><span /><strong>Clipping {clipping ? "on" : "off"}</strong></label></div>
        {gradient ? <MiniLineChart series={gradient.scenarios.map((scenario) => ({ name: scenario.scenario, values: scenario.points.map((point) => clipping ? point.gradient_norm_after : point.gradient_norm_before) }))} labels={gradient.scenarios[0].points.map((point) => String(point.length))} /> : <LoadingPanel label="Loading gradient experiment" />}
        {gradient && <StatusBanner kind="info">{gradient.interpretation}</StatusBanner>}
      </section>
      <StatusBanner kind="warning">Internal activations are observations of computation, not causal explanations of the activity label.</StatusBanner>
    </div>
  );
}
