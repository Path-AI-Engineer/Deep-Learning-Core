import {useEffect, useState} from "react";
import {api} from "../api/client";
import {Icon} from "../components/Icons";
import {Metric, PageHeader, Panel, PanelHeader, State, Status, TokenRail} from "../components/Primitives";
import type {Prediction, Sample, TaskId} from "../types/contracts";

export function TransductionPage() {
  const [task, setTask] = useState<TaskId>("copy");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selected, setSelected] = useState<Sample | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  useEffect(() => {api.samples(task).then(result => {setSamples(result.items); setSelected(result.items[0] ?? null); setPrediction(null);}).catch(reason => setError(String(reason)));}, [task]);
  async function run() {
    if (!selected) return;
    setRunning(true); setError(null);
    try {setPrediction(await api.predict(selected));} catch (reason) {setError(reason instanceof Error ? reason.message : "Inference failed.");} finally {setRunning(false);}
  }
  return <>
    <PageHeader eyebrow="Real greedy decoding" title="Watch the target emerge one token at a time." copy="Select a controlled sample, run the active bundle and compare EOS-bounded output with the oracle target. OOD lengths remain visibly marked." meta={<Status>CPU · batch 1</Status>} />
    {error && <State kind="error" title="Inference unavailable" copy={error} />}
    <Panel>
      <PanelHeader label="Input contract" title="Choose task and evidence" action={<button className="primary-button" disabled={!selected || running} onClick={run}>{running ? "Decoding…" : <>Run transduction <Icon name="arrow" /></>}</button>} />
      <div className="segmented" role="group" aria-label="Task">{(["copy", "reverse", "recall"] as TaskId[]).map(value => <button className={task === value ? "active" : ""} onClick={() => setTask(value)} key={value}>{value === "recall" ? "Associative recall" : value}</button>)}</div>
      <div className="sample-grid">{samples.map(sample => <button className={selected?.example_id === sample.example_id ? "sample-card selected" : "sample-card"} onClick={() => {setSelected(sample); setPrediction(null);}} key={sample.example_id}><span>{sample.split.replace("validation_", "")}</span><strong>{sample.content_length} {task === "recall" ? "pairs" : "symbols"}</strong><small>{sample.source_tokens.slice(1, 5).map(token => token.replace("SYMBOL_", "S")).join(" · ")}</small></button>)}</div>
      {selected && <div className="sequence-preview"><span>Source</span><TokenRail tokens={selected.source_tokens} /></div>}
    </Panel>
    {prediction ? <>
      {prediction.warning && <div className="warning-banner"><Status tone="warning">OOD length</Status><span>{prediction.warning}</span></div>}
      <div className="metric-grid four"><Metric label="Exact match" value={prediction.exact_match ? "Yes" : "No"} detail="Full EOS-normalized sequence" tone={prediction.exact_match ? "cyan" : "amber"} /><Metric label="Token accuracy" value={`${(prediction.token_accuracy * 100).toFixed(1)}%`} detail="PAD excluded" /><Metric label="Latency" value={`${prediction.latency_ms.toFixed(1)} ms`} detail="CPU batch size 1" /><Metric label="Stopped by" value={prediction.eos_status.toUpperCase()} detail={prediction.model_version} /></div>
      <Panel><PanelHeader label="Outcome" title="Prediction against oracle target" /><div className="sequence-comparison"><div><span>Prediction</span><TokenRail tokens={prediction.prediction} tone="prediction" /></div><div><span>Target</span><TokenRail tokens={prediction.target} tone="target" /></div></div></Panel>
      <Panel><PanelHeader label="Autoregressive timeline" title={`${prediction.decoding_steps.length} bounded steps`} /><div className="timeline">{prediction.decoding_steps.map(step => <div key={step.step}><span>{String(step.step).padStart(2, "0")}</span><i /><div><strong>Token ID {step.selected_token_id}</strong><small>{step.top_k.map(item => `${item.token_id}: ${(item.probability * 100).toFixed(1)}%`).join(" · ")}</small></div></div>)}</div></Panel>
    </> : <State kind="empty" title="No decoding run selected" copy="Choose a sample and run the active Transformer bundle." />}
  </>;
}

