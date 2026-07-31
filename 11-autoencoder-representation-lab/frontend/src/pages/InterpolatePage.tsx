import {useState} from "react";
import {api} from "../api/client";
import {Eyebrow, ImageFrame, Panel, State} from "../components/Primitives";
import {useResource} from "../hooks/useResource";
import type {Sample} from "../types/contracts";

type Interpolation={model_id:string;latent_distance:number;items:{alpha:number;image:string}[];warning:string};
export function InterpolatePage(){
  const samples=useResource(()=>api.get<{items:Sample[]}>("/samples?limit=30"));
  const [a,setA]=useState("");const [b,setB]=useState("");const [steps,setSteps]=useState(7);const [result,setResult]=useState<Interpolation|null>(null);const [error,setError]=useState("");
  const first=a||samples.data?.items[0]?.sample_id||"";const second=b||samples.data?.items[3]?.sample_id||"";
  async function run(){setError("");try{setResult(await api.post<Interpolation>("/latent/interpolate",{model_id:"conv-ae",sample_id_a:first,sample_id_b:second,steps}));}catch(reason){setError(reason instanceof Error?reason.message:"Interpolation failed");}}
  return <div className="page"><div className="page-title"><div><Eyebrow>Bounded latent path</Eyebrow><h1>Interpolate without pretending to sample.</h1><p>Encode two observed samples, connect their codes linearly and decode a limited number of intermediate states.</p></div></div><State loading={samples.loading} error={samples.error||error}/>
  {samples.data&&<><Panel className="inline-controls interpolation-controls"><label>Start<select value={first} onChange={e=>setA(e.target.value)}>{samples.data.items.map(item=><option key={item.sample_id} value={item.sample_id}>{item.class_name} · {item.sample_id}</option>)}</select></label><span className="flow-arrow">→</span><label>End<select value={second} onChange={e=>setB(e.target.value)}>{samples.data.items.map(item=><option key={item.sample_id} value={item.sample_id}>{item.class_name} · {item.sample_id}</option>)}</select></label><label>Steps<input type="range" min="3" max="12" value={steps} onChange={e=>setSteps(Number(e.target.value))}/><strong>{steps}</strong></label><button className="button primary" onClick={run}>Decode path</button></Panel>
  {result?<><div className="interpolation-grid">{result.items.map(item=><ImageFrame key={item.alpha} src={item.image} label={`α ${item.alpha.toFixed(2)}`} accent={item.alpha>0&&item.alpha<1}/>)}</div><Panel className="warning-panel"><strong>Latent distance {result.latent_distance.toFixed(3)}</strong><p>{result.warning} Plausible transitions do not establish a generative probability model.</p></Panel></>:<Panel className="empty-result compact"><h2>Choose two observed samples</h2><p>The endpoints use the same encoder, decoder and latent dimension.</p></Panel>}</>}</div>;
}
