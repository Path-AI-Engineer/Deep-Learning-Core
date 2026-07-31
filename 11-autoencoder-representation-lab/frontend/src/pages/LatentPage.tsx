import {useState} from "react";
import {api} from "../api/client";
import {LatentScatter} from "../components/LatentScatter";
import {Eyebrow, ImageFrame, Panel, State} from "../components/Primitives";
import {useResource} from "../hooks/useResource";
import type {LatentPoint} from "../types/contracts";

export function LatentPage(){
  const points=useResource(()=>api.get<{items:LatentPoint[];bounds:{x:number[];y:number[]};warning:string}>("/latent/points?limit=200"));
  const [selected,setSelected]=useState<LatentPoint|null>(null);
  return <div className="page"><div className="page-title"><div><Eyebrow>Direct two-dimensional bottleneck</Eyebrow><h1>Navigate the learned surface.</h1><p>No t-SNE or UMAP: every point is the actual output of the independently trained 2D encoder.</p></div></div><State loading={points.loading} error={points.error} empty={points.data?.items.length===0}/>
  {points.data&&<div className="latent-layout"><Panel className="scatter-panel"><div className="panel-heading"><div><h2>Evaluation embeddings</h2><p>{points.data.items.length} controlled points</p></div><span className="axis-bounds">x [{points.data.bounds.x.map(v=>v.toFixed(2)).join(", ")}]<br/>y [{points.data.bounds.y.map(v=>v.toFixed(2)).join(", ")}]</span></div><LatentScatter points={points.data.items} onSelect={setSelected}/></Panel><Panel className="point-inspector">{selected?<><Eyebrow>Selected coordinate</Eyebrow><h2>{selected.class_name}</h2><code>({selected.x.toFixed(4)}, {selected.y.toFixed(4)})</code><div className="paired-images"><ImageFrame src={selected.image} label="Original"/><ImageFrame src={selected.reconstruction} label="2D reconstruction" accent/></div><h3>Nearest observed codes</h3><ol className="neighbor-list">{selected.neighbors.map(item=><li key={item.sample_id}><span>{item.sample_id}</span><strong>{item.distance.toFixed(3)}</strong></li>)}</ol></>:<><span className="empty-orbit small"/><h2>Select a point</h2><p>Inspect its image, reconstruction and Euclidean neighbors. Proximity is exploratory—not proof of semantic identity.</p></>}</Panel></div>}</div>;
}
