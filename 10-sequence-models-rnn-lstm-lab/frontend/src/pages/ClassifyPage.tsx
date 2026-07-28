import { useEffect, useState } from "react";
import { api } from "../api/client";
import { SignalChart } from "../components/Charts";
import { ActivityIcon, ArrowIcon } from "../components/Icons";
import {
  EmptyPanel,
  ErrorPanel,
  LoadingPanel,
  PageHeader,
  StatusBanner,
  formatPercent
} from "../components/Primitives";
import type { ModelId, Prediction, SampleDetail, SampleSummary } from "../types/contracts";

const activities = ["ALL", "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"];

export function ClassifyPage({ navigate }: { navigate: (path: string) => void }) {
  const [samples, setSamples] = useState<SampleSummary[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState<SampleDetail | null>(null);
  const [activity, setActivity] = useState("ALL");
  const [model, setModel] = useState<ModelId>("rnn");
  const [channels, setChannels] = useState([0, 1, 3]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [comparison, setComparison] = useState<Prediction[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let ignore = false;
    setLoading(true);
    api.samples(activity === "ALL" ? undefined : activity)
      .then((rows) => {
        if (ignore) return;
        setSamples(rows);
        setSelectedId(rows[0]?.sample_id ?? "");
      })
      .catch((reason: Error) => !ignore && setError(reason.message))
      .finally(() => !ignore && setLoading(false));
    return () => { ignore = true; };
  }, [activity]);

  useEffect(() => {
    if (!selectedId) { setDetail(null); return; }
    let ignore = false;
    api.sample(selectedId)
      .then((result) => !ignore && setDetail(result))
      .catch((reason: Error) => !ignore && setError(reason.message));
    setPrediction(null);
    setComparison(null);
    return () => { ignore = true; };
  }, [selectedId]);

  async function classify() {
    if (!selectedId) return;
    setRunning(true);
    setError(null);
    try {
      setPrediction(await api.predict(selectedId, model));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Classification failed");
    } finally {
      setRunning(false);
    }
  }

  async function compare() {
    if (!selectedId) return;
    setRunning(true);
    try {
      setComparison((await api.comparePrediction(selectedId)).predictions);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Comparison failed");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="page">
      <PageHeader eyebrow="Controlled inference" title="Read an activity through time." description="Choose a bounded test fixture, inspect all nine channels and execute the exact same sequence through recurrent models." />
      {error && <ErrorPanel message={error} />}
      <div className="filter-bar">
        <label>Activity<select value={activity} onChange={(event) => setActivity(event.target.value)}>{activities.map((item) => <option key={item}>{item}</option>)}</select></label>
        <label>Model<select value={model} onChange={(event) => setModel(event.target.value as ModelId)}>{["rnn", "lstm", "gru"].map((item) => <option key={item} value={item}>{item.toUpperCase()}</option>)}</select></label>
        <div className="filter-meta"><span>{samples.length}</span> controlled samples</div>
      </div>
      {loading && <LoadingPanel label="Loading sequence gallery" />}
      {!loading && samples.length === 0 && <EmptyPanel title="No samples match this filter" message="Choose another activity to continue." />}
      {samples.length > 0 && (
        <div className="classifier-layout">
          <aside className="sample-rail" aria-label="Sample gallery">
            {samples.map((sample) => (
              <button key={sample.sample_id} className={selectedId === sample.sample_id ? "sample-card active" : "sample-card"} onClick={() => setSelectedId(sample.sample_id)}>
                <span className="sample-icon"><ActivityIcon /></span>
                <span><strong>{sample.activity.replaceAll("_", " ")}</strong><small>{sample.sample_id} · {sample.subject_id}</small></span>
              </button>
            ))}
          </aside>
          <section className="sequence-workbench">
            {detail ? (
              <>
                <div className="workbench-heading">
                  <div><p className="eyebrow">128 timesteps · 50 Hz</p><h2>{detail.activity.replaceAll("_", " ")}</h2><p>{detail.sample_id} · {detail.subject_id}</p></div>
                  <span className="shape-chip">[1, 128, 9]</span>
                </div>
                <div className="channel-picker" aria-label="Visible signal channels">
                  {detail.channels.map((name, index) => (
                    <button key={name} className={channels.includes(index) ? "channel active" : "channel"} onClick={() => setChannels((current) => current.includes(index) ? current.filter((value) => value !== index) : current.length < 5 ? [...current, index] : current)}>{name}</button>
                  ))}
                </div>
                {channels.length ? <SignalChart signals={detail.signals} channels={detail.channels} selected={channels} /> : <EmptyPanel title="No channel selected" message="Select up to five channels to draw the temporal signal." />}
                <div className="action-row">
                  <button className="button button-primary" onClick={classify} disabled={running}>{running ? "Running inference…" : `Classify with ${model.toUpperCase()}`} <ArrowIcon /></button>
                  <button className="button button-secondary" onClick={compare} disabled={running}>Compare all models</button>
                </div>
              </>
            ) : <LoadingPanel label="Loading full signal tensor" />}
          </section>
        </div>
      )}
      {prediction && (
        <section className="prediction-panel">
          <div className="prediction-verdict"><p className="eyebrow">Inference result</p><span>Predicted activity</span><h2>{prediction.predicted_class.replaceAll("_", " ")}</h2><p>Ground truth: <strong>{prediction.true_class.replaceAll("_", " ")}</strong></p></div>
          <div className="probability-stack">{prediction.top_k.map((item) => <div key={item.class_name}><span>{item.class_name.replaceAll("_", " ")}</span><div><i style={{ width: `${item.probability * 100}%` }} /></div><strong>{formatPercent(item.probability)}</strong></div>)}</div>
          <div className="prediction-meta"><span><small>Model</small>{prediction.model_type.toUpperCase()} · {prediction.model_version}</span><span><small>Latency</small>{prediction.latency_ms.toFixed(2)} ms</span><button className="text-link" onClick={() => navigate("/sequence-lab")}>Inspect memory trace <ArrowIcon /></button></div>
          <StatusBanner kind="info">{prediction.warnings[0]}</StatusBanner>
        </section>
      )}
      {comparison && <section className="panel"><div className="panel-heading"><div><p className="eyebrow">Same sequence, three memories</p><h2>Model agreement</h2></div></div><div className="comparison-cards">{comparison.map((item) => <article key={item.model_type}><span>{item.model_type.toUpperCase()}</span><strong>{item.predicted_class.replaceAll("_", " ")}</strong><small>{formatPercent(item.confidence)} confidence · {item.latency_ms.toFixed(2)} ms</small></article>)}</div></section>}
    </div>
  );
}
