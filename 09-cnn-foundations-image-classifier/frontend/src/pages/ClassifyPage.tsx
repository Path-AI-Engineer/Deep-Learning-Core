import { useEffect, useRef, useState } from "react";
import { Icon } from "../components/Icons";
import { EmptyState, ErrorBanner, PageHeader, Panel } from "../components/Primitives";
import { api, type Prediction, type Sample } from "../lib/api";

function Result({ prediction }: { prediction: Prediction }) {
  return (
    <Panel eyebrow="MODEL OUTPUT" title={prediction.predicted_class} className="result-panel">
      <div className="prediction-lead">
        <img src={prediction.preprocessed_preview} alt="Preprocessed 28 by 28 grayscale input"/>
        <div><span>Top prediction</span><strong>{(prediction.top_k[0].probability * 100).toFixed(1)}%</strong><small>{prediction.inference_time_ms.toFixed(1)} ms · {prediction.model_version}</small></div>
      </div>
      <div className="probability-list">
        {prediction.top_k.map((row) => <div key={row.index}><span>{row.class_name}</span><div><i style={{ width: `${row.probability * 100}%` }}/></div><strong>{(row.probability * 100).toFixed(1)}%</strong></div>)}
      </div>
      {prediction.warnings.map((warning) => <p className="model-warning" key={warning}>{warning}</p>)}
    </Panel>
  );
}

export function ClassifyPage() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.samples().then((result) => { setSamples(result.items); setSelected(result.items[0]?.sample_id ?? ""); }).catch((reason: Error) => setError(reason.message));
  }, []);

  async function runSample() {
    if (!selected) return;
    setBusy(true); setError("");
    try { setPrediction(await api.predictSample(selected)); } catch (reason) { setError((reason as Error).message); } finally { setBusy(false); }
  }

  async function runUpload(file?: File) {
    if (!file) return;
    setBusy(true); setError("");
    try { setPrediction(await api.predictUpload(file)); } catch (reason) { setError((reason as Error).message); } finally { setBusy(false); }
  }

  return (
    <>
      <PageHeader eyebrow="CONTROLLED INFERENCE" title="Classify an image." description="Choose an official held-out sample or inspect how the model responds to a PNG or JPEG processed entirely in memory."/>
      {error && <ErrorBanner message={error}/>}
      <div className="split-layout">
        <Panel eyebrow="INPUT SOURCE" title="Select evidence">
          <div className="source-tabs"><button className="active">Test gallery</button><button onClick={() => input.current?.click()}>Upload image</button></div>
          {samples.length ? <div className="sample-grid">
            {samples.map((sample) => <button aria-label={`${sample.class_name}, ${sample.sample_id}`} className={selected === sample.sample_id ? "sample active" : "sample"} key={sample.sample_id} onClick={() => setSelected(sample.sample_id)}><img alt="" src={sample.image_data_url}/><span>{sample.class_name}</span></button>)}
          </div> : <EmptyState title="Gallery unavailable">Prepare the official FashionMNIST dataset to enable held-out samples.</EmptyState>}
          <input accept=".png,.jpg,.jpeg,image/png,image/jpeg" hidden onChange={(event) => runUpload(event.target.files?.[0])} ref={input} type="file"/>
          <div className="action-row">
            <button className="secondary-button" onClick={() => input.current?.click()}><Icon name="upload" size={18}/> Upload</button>
            <button className="primary-button" disabled={!selected || busy} onClick={runSample}>{busy ? "Running…" : "Run classification"} <Icon name="arrow" size={18}/></button>
          </div>
        </Panel>
        {prediction ? <Result prediction={prediction}/> : <Panel eyebrow="MODEL OUTPUT" title="Prediction"><EmptyState title="No inference yet">Choose a sample and run classification to inspect calibrated probabilities, preprocessing and runtime evidence.</EmptyState></Panel>}
      </div>
    </>
  );
}
