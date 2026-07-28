import { useState } from "react";
import { ErrorBanner, PageHeader, Panel } from "../components/Primitives";
import { api } from "../lib/api";

const matrix = [
  [0, 0, 0, 1, 1, 0],
  [0, 0, 1, 1, 0, 0],
  [0, 1, 1, 0, 0, 0],
  [1, 1, 0, 0, 0, 0],
  [1, 0, 0, 0, 1, 1],
  [0, 0, 0, 1, 1, 0]
];

const kernels: Record<string, number[][]> = {
  "Vertical edge": [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
  "Horizontal edge": [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
  Sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
};

function NumberGrid({ values, heat = false }: { values: number[][]; heat?: boolean }) {
  const maximum = Math.max(...values.flat().map(Math.abs), 1);
  return <div className="number-grid" style={{ gridTemplateColumns: `repeat(${values[0]?.length ?? 1}, minmax(32px, 1fr))` }}>{values.flatMap((row, y) => row.map((value, x) => <span key={`${y}-${x}`} style={heat ? { background: `rgba(49, 216, 192, ${0.08 + Math.abs(value) / maximum * 0.68})` } : undefined}>{Number(value.toFixed(1))}</span>))}</div>;
}

export function ConvolutionPage() {
  const [name, setName] = useState("Vertical edge");
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(1);
  const [output, setOutput] = useState<number[][]>([]);
  const [parity, setParity] = useState<{ passed: boolean; max_absolute_error: number } | null>(null);
  const [error, setError] = useState("");

  async function calculate() {
    setError("");
    try {
      const result = await api.convolution(matrix, kernels[name], stride, padding);
      setOutput(result.output); setParity(result.parity_result);
    } catch (reason) { setError((reason as Error).message); }
  }

  return (
    <>
      <PageHeader eyebrow="SPATIAL OPERATIONS" title="Watch a kernel move." description="Compute two-dimensional cross-correlation by hand and confirm every output against torch.nn.functional.conv2d."/>
      {error && <ErrorBanner message={error}/>}
      <div className="lab-controls panel">
        <label>Kernel<select value={name} onChange={(event) => setName(event.target.value)}>{Object.keys(kernels).map((value) => <option key={value}>{value}</option>)}</select></label>
        <label>Stride<select value={stride} onChange={(event) => setStride(Number(event.target.value))}><option value={1}>1 pixel</option><option value={2}>2 pixels</option></select></label>
        <label>Padding<select value={padding} onChange={(event) => setPadding(Number(event.target.value))}><option value={0}>None</option><option value={1}>1 pixel</option></select></label>
        <button className="primary-button" onClick={calculate}>Calculate output</button>
      </div>
      <div className="convolution-grid">
        <Panel eyebrow="INPUT · 6 × 6" title="Image matrix"><NumberGrid values={matrix}/></Panel>
        <Panel eyebrow="KERNEL · 3 × 3" title={name}><NumberGrid values={kernels[name]}/><p className="formula">output[y,x] = Σ input[y+i,x+j] × kernel[i,j]</p></Panel>
        <Panel eyebrow={output.length ? `OUTPUT · ${output.length} × ${output[0].length}` : "OUTPUT"} title="Feature response">{output.length ? <NumberGrid values={output} heat/> : <div className="grid-placeholder">Run the operation to reveal the response map.</div>}{parity && <p className={parity.passed ? "parity parity--pass" : "parity"}>{parity.passed ? "PyTorch parity passed" : "Parity failed"} · max error {parity.max_absolute_error.toExponential(1)}</p>}</Panel>
      </div>
    </>
  );
}
