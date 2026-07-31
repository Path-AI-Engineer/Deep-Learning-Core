import {useState} from "react";
import {api} from "../api/client";
import {Heatmap} from "../components/Heatmap";
import {PageHeader, Panel, PanelHeader, State, Status} from "../components/Primitives";

const PRESET = {
  query: [[1, 0], [0, 1], [1, 1]],
  key: [[1, 0], [0, 1], [1, 1]],
  value: [[1, 2], [3, 1], [2, 4]],
};

export function AttentionMathPage() {
  const [masked, setMasked] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  async function calculate() {
    setRunning(true); setError(null);
    try {
      setResult(await api.attention({...PRESET, mask: masked ? [[false, true, true], [false, false, true], [false, false, false]] : undefined}));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Calculation failed.");
    } finally {setRunning(false);}
  }
  const weights = result?.weights as number[][] | undefined;
  const output = result?.output as number[][] | undefined;
  return <>
    <PageHeader eyebrow="Scaled dot-product attention" title="Audit every operation between Q and context." copy="The API computes QKᵀ, scales by √dₖ, applies the selected mask, normalizes across keys and combines V. Nothing is animated without a corresponding value." meta={<Status tone="neutral">Educational fixture · float64</Status>} />
    {error && <State kind="error" title="Attention calculation rejected" copy={error} />}
    <div className="math-layout">
      <Panel>
        <PanelHeader label="Input matrices" title="Q, K and V" action={<button className="primary-button compact" disabled={running} onClick={calculate}>{running ? "Computing…" : "Compute attention"}</button>} />
        <div className="matrix-trio">{Object.entries(PRESET).map(([name, matrix]) => <div className="mini-matrix" key={name}><strong>{name.toUpperCase()}</strong>{matrix.map((row, index) => <span key={index}>[{row.join(", ")}]</span>)}</div>)}</div>
        <label className="toggle-row"><input type="checkbox" checked={masked} onChange={event => setMasked(event.target.checked)} /><span><strong>Apply causal mask</strong><small>True means blocked before softmax.</small></span></label>
        <div className="formula"><span>Attention(Q, K, V)</span><strong>= softmax((QKᵀ / √dₖ) + M) V</strong></div>
      </Panel>
      <Panel>
        <PanelHeader label="Normalization audit" title="Attention weights" action={result && <Status>Difference {Number(result.reference_difference).toExponential(1)}</Status>} />
        {weights ? <Heatmap matrix={weights} rows={["q0", "q1", "q2"]} columns={["k0", "k1", "k2"]} label="Computed attention weight matrix" /> : <State kind="empty" title="No calculation yet" copy="Run the bounded fixture to inspect normalized weights." />}
      </Panel>
    </div>
    {output && <Panel><PanelHeader label="Weighted combination" title="Context vectors" /><div className="output-vectors">{output.map((row, index) => <div key={index}><span>query {index}</span><strong>[{row.map(value => value.toFixed(4)).join(", ")}]</strong></div>)}</div></Panel>}
  </>;
}

