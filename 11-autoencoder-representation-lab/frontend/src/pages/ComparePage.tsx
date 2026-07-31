import {api} from "../api/client";
import {Eyebrow, Panel, Pill, State} from "../components/Primitives";
import {useResource} from "../hooks/useResource";
import type {ModelRow} from "../types/contracts";

export function ComparePage(){
  const data=useResource(()=>api.get<{active_model:string;selection:string;models:ModelRow[];warning:string}>("/evaluation/summary"));
  return <div className="page"><div className="page-title"><div><Eyebrow>Multi-dimensional model decision</Eyebrow><h1>Keep conflicting evidence visible.</h1><p>Reconstruction, probe utility, robustness, capacity and operational capability answer different questions.</p></div></div><State loading={data.loading} error={data.error}/>
  {data.data&&<><Panel className="decision-card"><div><Pill tone="good">Active neural bundle</Pill><h2>{data.data.active_model}</h2></div><p>{data.data.selection}</p></Panel><Panel className="table-panel"><div className="responsive-table"><table><thead><tr><th>Representation</th><th>Latent</th><th>MSE</th><th>SSIM</th><th>Probe F1</th><th>Parameters</th></tr></thead><tbody>{data.data.models.map(model=><tr key={model.model_id} className={model.model_id===data.data?.active_model?"selected":""}><td><strong>{model.model_id}</strong>{model.model_id===data.data?.active_model&&<Pill tone="good">active</Pill>}</td><td>{model.model_id==="mean-image"?"—":model.model_id==="latent-2d"?"2":"16"}</td><td>{model.reconstruction.mse.toFixed(4)}</td><td>{model.reconstruction.ssim.toFixed(3)}</td><td>{model.representation.linear_probe?.macro_f1.toFixed(3)??"N/A"}</td><td>{model.parameters.toLocaleString()}</td></tr>)}</tbody></table></div></Panel><p className="footnote">{data.data.warning}</p></>}</div>;
}
