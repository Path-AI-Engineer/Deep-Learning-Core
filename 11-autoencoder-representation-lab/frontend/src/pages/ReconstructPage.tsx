import {useState} from "react";
import {api} from "../api/client";
import {Eyebrow, ImageFrame, MetricStrip, Panel, State} from "../components/Primitives";
import {useResource} from "../hooks/useResource";
import type {ModelRow, Reconstruction, Sample} from "../types/contracts";

export function ReconstructPage() {
  const samples=useResource(()=>api.get<{items:Sample[]}>("/samples?limit=30"));
  const models=useResource(()=>api.get<{items:ModelRow[]}>("/models"));
  const [sampleId,setSampleId]=useState("");
  const [modelId,setModelId]=useState("conv-ae");
  const [result,setResult]=useState<Reconstruction|null>(null);
  const [error,setError]=useState("");
  const [busy,setBusy]=useState(false);
  const selected=sampleId || samples.data?.items[0]?.sample_id || "";
  async function run() { setBusy(true); setError(""); try { setResult(await api.post<Reconstruction>("/reconstruct/sample",{sample_id:selected,model_id:modelId})); } catch(reason){setError(reason instanceof Error?reason.message:"Reconstruction failed");} finally{setBusy(false);} }
  async function upload(file:File|undefined) { if(!file)return; setBusy(true); setError(""); try { setResult(await api.upload<Reconstruction>(`/reconstruct/upload?model_id=${encodeURIComponent(modelId)}`,file)); } catch(reason){setError(reason instanceof Error?reason.message:"Upload reconstruction failed");} finally{setBusy(false);} }
  return <div className="page"><div className="page-title"><div><Eyebrow>Reconstruction studio</Eyebrow><h1>Inspect what the code preserves.</h1><p>Compare the original, decoded output and absolute error without treating visual appeal as sufficient evidence.</p></div></div>
    <State loading={samples.loading||models.loading} error={samples.error||models.error||error}/>
    {samples.data&&models.data&&<div className="lab-layout">
      <Panel className="controls"><label htmlFor="sample">Controlled test sample</label><select id="sample" value={selected} onChange={e=>setSampleId(e.target.value)}>{samples.data.items.map(item=><option key={item.sample_id} value={item.sample_id}>{item.class_name} · {item.sample_id}</option>)}</select><label htmlFor="model">Representation</label><select id="model" value={modelId} onChange={e=>setModelId(e.target.value)}>{models.data.items.filter(item=>item.model_id!=="latent-2d").map(item=><option key={item.model_id}>{item.model_id}</option>)}</select><button className="button primary wide" disabled={busy||!selected} onClick={run}>{busy?"Reconstructing…":"Run reconstruction"}</button><label className="upload-control" htmlFor="upload">Or use an ephemeral image<input id="upload" type="file" accept="image/png,image/jpeg,image/webp" disabled={busy||modelId==="mean-image"||modelId==="pca"} onChange={event=>void upload(event.target.files?.[0])}/><span>PNG, JPEG or WebP · maximum 1 MB</span></label><div className="control-note"><strong>No labels enter the model.</strong><span>Uploads are decoded in memory and are never retained.</span></div></Panel>
      <div className="result-area">{!result?<Panel className="empty-result"><span className="empty-orbit"/><h2>Choose a representation</h2><p>Run one controlled sample to reveal its reconstruction and per-image metrics.</p></Panel>:<><div className="image-triptych"><ImageFrame src={result.original} label={`Original · ${result.class_name}`}/><ImageFrame src={result.reconstruction} label={`${result.model_id} reconstruction`} accent/><ImageFrame src={result.absolute_error} label="Absolute error map"/></div><MetricStrip metrics={result.metrics}/><Panel className="latent-readout"><div><Eyebrow>Bottleneck</Eyebrow><h3>{result.latent.length||"Linear"} dimensions exposed</h3></div><code>{result.latent.length?result.latent.slice(0,8).map(value=>value.toFixed(3)).join(" · "):"Baseline has no neural latent code"}</code></Panel></>}</div>
    </div>}</div>;
}
