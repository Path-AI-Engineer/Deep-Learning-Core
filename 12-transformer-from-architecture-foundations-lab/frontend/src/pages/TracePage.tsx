import {useEffect, useState} from "react";
import {api} from "../api/client";
import {Heatmap} from "../components/Heatmap";
import {PageHeader, Panel, PanelHeader, State, Status, TokenRail} from "../components/Primitives";
import type {Sample, TaskId, Trace, TraceType} from "../types/contracts";

export function TracePage({explorer = false}: {explorer?: boolean}) {
  const [task, setTask] = useState<TaskId>("reverse");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [sample, setSample] = useState<Sample | null>(null);
  const [type, setType] = useState<TraceType>(explorer ? "cross" : "encoder_self");
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  useEffect(() => {api.samples(task).then(result => {setSamples(result.items); setSample(result.items[0] ?? null); setTrace(null);});}, [task]);
  async function run() {
    if (!sample) return;
    setRunning(true); setError(null);
    try {setTrace(await api.trace(sample, type, layer, head));} catch (reason) {setError(reason instanceof Error ? reason.message : "Trace failed.");} finally {setRunning(false);}
  }
  return <>
    <PageHeader eyebrow={explorer ? "Attention explorer" : "Bounded architecture trace"} title={explorer ? "Inspect a head without pretending it explains the model." : "Move through the forward path with real tensors."} copy={explorer ? "Choose attention type, layer and head. Every heatmap preserves token axes, numerical values and an accessible table." : "The trace captures one sample, one approved layer/head selection and a bounded subset of values from actual inference."} meta={<Status tone="neutral">Trace schema 1.0</Status>} />
    {error && <State kind="error" title="Trace unavailable" copy={error} />}
    <Panel>
      <PanelHeader label="Trace controls" title="Select evidence boundary" action={<button className="primary-button compact" disabled={!sample || running} onClick={run}>{running ? "Tracing…" : "Run trace"}</button>} />
      <div className="trace-controls">
        <label>Task<select value={task} onChange={event => setTask(event.target.value as TaskId)}><option value="copy">Copy</option><option value="reverse">Reverse</option><option value="recall">Associative recall</option></select></label>
        <label>Sample<select value={sample?.example_id ?? ""} onChange={event => setSample(samples.find(item => item.example_id === event.target.value) ?? null)}>{samples.map(item => <option value={item.example_id} key={item.example_id}>{item.split} · {item.content_length}</option>)}</select></label>
        <label>Attention<select value={type} onChange={event => setType(event.target.value as TraceType)}><option value="encoder_self">Encoder self</option><option value="decoder_self">Decoder self</option><option value="cross">Cross attention</option></select></label>
        <label>Layer<select value={layer} onChange={event => setLayer(Number(event.target.value))}><option value="0">Layer 0</option><option value="1">Layer 1</option></select></label>
        <label>Head<select value={head} onChange={event => setHead(Number(event.target.value))}>{[0,1,2,3].map(value => <option value={value} key={value}>Head {value}</option>)}</select></label>
      </div>
      {sample && <TokenRail tokens={sample.source_tokens} />}
    </Panel>
    {trace ? <div className="trace-layout">
      <Panel className="heatmap-panel"><PanelHeader label={`${trace.trace_type.replace("_", " ")} · layer ${trace.layer} · head ${trace.head}`} title="Attention weight matrix" action={<Status>Shape {trace.shape.join(" × ")}</Status>} /><Heatmap matrix={trace.weights} rows={trace.query_tokens} columns={trace.key_tokens} label={`${trace.trace_type} attention weights`} /></Panel>
      <Panel><PanelHeader label="Descriptor" title="Entropy by query" /><div className="entropy-list">{trace.entropy.map((value, index) => <div key={index}><span>{trace.query_tokens[index] ?? `q${index}`}</span><div><i style={{width: `${Math.min(100, value / 3 * 100)}%`}} /></div><strong>{value.toFixed(3)}</strong></div>)}</div><div className="warning-card"><Status tone="warning">Interpretation boundary</Status><p>{trace.warning}</p></div></Panel>
    </div> : <State kind="empty" title="No trace captured" copy="Run a bounded trace to connect a real attention matrix with its token axes." />}
  </>;
}

