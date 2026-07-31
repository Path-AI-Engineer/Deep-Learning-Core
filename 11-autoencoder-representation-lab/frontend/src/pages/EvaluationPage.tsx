import {useState} from "react";
import {api} from "../api/client";
import {Eyebrow, ImageFrame, Panel, State} from "../components/Primitives";
import {useResource} from "../hooks/useResource";
import type {ModelRow} from "../types/contracts";

type ErrorRow={sample_id:string;class_name:string;mse:number;original:string;reconstruction:string};
export function EvaluationPage(){
  const models=useResource(()=>api.get<{items:ModelRow[]}>("/models"));
  const [model,setModel]=useState("conv-ae");
  const errors=useResource(()=>api.get<{items:ErrorRow[]}>(`/evaluation/errors?model_id=${model}&limit=8`),[model]);
  return <div className="page"><div className="page-title split"><div><Eyebrow>Evaluation & error analysis</Eyebrow><h1>Study where reconstruction fails.</h1><p>Worst-case views complement averages and expose smoothing, missing detail and representation tradeoffs.</p></div><label className="title-filter">Model<select value={model} onChange={e=>setModel(e.target.value)}>{models.data?.items.map(item=><option key={item.model_id}>{item.model_id}</option>)}</select></label></div><State loading={models.loading||errors.loading} error={models.error||errors.error}/>
  {errors.data&&<div className="error-grid">{errors.data.items.map((item,index)=><Panel key={item.sample_id} className="error-card"><div className="error-rank">#{index+1}<span>{item.class_name}</span><strong>MSE {item.mse.toFixed(4)}</strong></div><div className="paired-images"><ImageFrame src={item.original} label="Original"/><ImageFrame src={item.reconstruction} label="Reconstruction" accent/></div></Panel>)}</div>}</div>;
}
