import { useState } from "react";
import { EmptyState, ErrorBanner, PageHeader, Panel } from "../components/Primitives";
import { api } from "../lib/api";

function Heatmap({ values }: { values: number[][] }) {
  return <div className="heatmap" style={{ gridTemplateColumns: `repeat(${values[0]?.length ?? 1}, 1fr)` }}>{values.flatMap((row, y) => row.map((value, x) => <i key={`${y}-${x}`} style={{ background: `rgb(${Math.round(9 + value * 50)}, ${Math.round(26 + value * 200)}, ${Math.round(30 + value * 170)})` }}/>))}</div>;
}

export function FeatureMapsPage() {
  const [sampleId, setSampleId] = useState("test-00000");
  const [layer, setLayer] = useState("conv1");
  const [result, setResult] = useState<Awaited<ReturnType<typeof api.activations>> | null>(null);
  const [error, setError] = useState("");

  async function inspect() {
    setError("");
    try { setResult(await api.activations(sampleId, layer)); } catch (reason) { setError((reason as Error).message); }
  }

  return (
    <>
      <PageHeader eyebrow="CONTROLLED ACTIVATIONS" title="Inspect intermediate features." description="Capture a whitelisted layer for one held-out sample. These maps are recorded activations—not causal explanations."/>
      {error && <ErrorBanner message={error}/>}
      <div className="lab-controls panel">
        <label>Sample ID<input value={sampleId} onChange={(event) => setSampleId(event.target.value)}/></label>
        <label>Layer<select value={layer} onChange={(event) => setLayer(event.target.value)}><option>conv1</option><option>pool1</option><option>conv2</option><option>pool2</option></select></label>
        <button className="primary-button" onClick={inspect}>Capture activations</button>
      </div>
      <Panel eyebrow={result ? `${result.layer_id.toUpperCase()} · ${result.tensor_shape.join(" × ")}` : "FEATURE MAPS"} title="Spatial responses">
        {result ? <><div className="feature-map-grid">{result.feature_maps.map((values, index) => <figure key={index}><Heatmap values={values}/><figcaption>Channel {String(index + 1).padStart(2, "0")}</figcaption></figure>)}</div><p className="model-warning">{result.interpretation_warning}</p></> : <EmptyState title="No captured layer">Enter a controlled test ID and select one of the approved layers.</EmptyState>}
      </Panel>
    </>
  );
}
