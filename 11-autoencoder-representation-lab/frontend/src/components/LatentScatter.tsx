import type {LatentPoint} from "../types/contracts";

const colors=["#69e4bd","#f7b955","#71a9ff","#f68181","#c7a0ff","#5dd7e5","#ff8dc7","#9fda77","#e6d873","#ff9d63"];
export function LatentScatter({points,onSelect}:{points:LatentPoint[];onSelect:(point:LatentPoint)=>void}) {
  if (!points.length) return null;
  const xs=points.map(p=>p.x), ys=points.map(p=>p.y);
  const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const px=(x:number)=>32+((x-minX)/(maxX-minX||1))*636;
  const py=(y:number)=>328-((y-minY)/(maxY-minY||1))*296;
  return <div className="scatter-wrap">
    <svg viewBox="0 0 700 360" role="img" aria-label={`Two-dimensional latent scatter with ${points.length} evaluation samples`}>
      <g className="grid-lines">{[0,1,2,3,4].map(i=><line key={`v${i}`} x1={32+i*159} y1="20" x2={32+i*159} y2="328"/>)}{[0,1,2,3,4].map(i=><line key={`h${i}`} x1="32" y1={20+i*77} x2="668" y2={20+i*77}/>)}</g>
      {points.map(point=><g key={point.sample_id} role="button" tabIndex={0} aria-label={`${point.class_name}, ${point.sample_id}`} onClick={()=>onSelect(point)} onKeyDown={event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();onSelect(point);}}}><circle cx={px(point.x)} cy={py(point.y)} r="6" fill={colors[point.label]} stroke="#07120f" strokeWidth="2"/></g>)}
    </svg>
    <p className="chart-summary">Each point is a direct 2D bottleneck coordinate. Labels color evaluation only; they were not used in autoencoder training.</p>
  </div>;
}
