import {useState} from "react";
import {api} from "../api/client";
import {Eyebrow, ImageFrame, MetricStrip, Panel, State} from "../components/Primitives";
import {useResource} from "../hooks/useResource";
import type {Metrics, Sample} from "../types/contracts";

type DenoiseResult={clean:string;corrupted:string;corruption:{type:string;level:number;seed:number};reconstructions:{model_id:string;image:string;metrics:Metrics}[];warning:string};
export function DenoisePage(){
  const samples=useResource(()=>api.get<{items:Sample[]}>("/samples?limit=30"));
  const [sampleId,setSampleId]=useState(""); const [kind,setKind]=useState("gaussian"); const [level,setLevel]=useState(0.2); const [seed,setSeed]=useState(42);
  const [result,setResult]=useState<DenoiseResult|null>(null); const [error,setError]=useState(""); const [busy,setBusy]=useState(false);
  const selected=sampleId||samples.data?.items[0]?.sample_id||"";
  async function run(){setBusy(true);setError("");try{setResult(await api.post<DenoiseResult>("/denoise",{sample_id:selected,corruption_type:kind,corruption_level:level,seed,model_ids:["conv-ae","denoising-ae"]}));}catch(reason){setError(reason instanceof Error?reason.message:"Denoising failed");}finally{setBusy(false);}}
  return <div className="page"><div className="page-title"><div><Eyebrow>Matched corruption protocol</Eyebrow><h1>Recover signal, not noise.</h1><p>Both models see the same seeded corruption and are evaluated against the untouched clean image.</p></div></div><State loading={samples.loading} error={samples.error||error}/>
  {samples.data&&<><Panel className="inline-controls"><label>Sample<select value={selected} onChange={e=>setSampleId(e.target.value)}>{samples.data.items.map(item=><option key={item.sample_id} value={item.sample_id}>{item.class_name} · {item.sample_id}</option>)}</select></label><label>Corruption<select value={kind} onChange={e=>setKind(e.target.value)}><option value="gaussian">Gaussian noise</option><option value="masking">Masking noise</option></select></label><label>Level<select value={level} onChange={e=>setLevel(Number(e.target.value))}><option value=".1">0.10</option><option value=".2">0.20</option><option value=".3">0.30</option></select></label><label>Seed<select value={seed} onChange={e=>setSeed(Number(e.target.value))}><option>7</option><option>21</option><option>42</option></select></label><button className="button primary" disabled={busy} onClick={run}>{busy?"Evaluating…":"Compare recovery"}</button></Panel>
  {!result?<Panel className="empty-result compact"><h2>Set one controlled corruption</h2><p>The clean target is retained throughout evaluation.</p></Panel>:<div className="denoise-grid"><div><ImageFrame src={result.clean} label="Clean target"/><ImageFrame src={result.corrupted} label={`${result.corruption.type} · ${result.corruption.level}`}/></div>{result.reconstructions.map(item=><Panel key={item.model_id} className="denoise-result"><ImageFrame src={item.image} label={item.model_id} accent={item.model_id==="denoising-ae"}/><MetricStrip metrics={item.metrics}/></Panel>)}</div>}</>}</div>;
}
