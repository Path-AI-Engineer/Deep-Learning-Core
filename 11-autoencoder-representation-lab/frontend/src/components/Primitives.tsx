import type {ReactNode} from "react";
import type {Metrics} from "../types/contracts";

export function Eyebrow({children}:{children:ReactNode}) { return <div className="eyebrow">{children}</div>; }
export function Pill({children,tone="neutral"}:{children:ReactNode;tone?:"neutral"|"good"|"warn"}) { return <span className={`pill ${tone}`}>{children}</span>; }
export function Panel({children,className=""}:{children:ReactNode;className?:string}) { return <section className={`panel ${className}`}>{children}</section>; }
export function State({loading,error,empty=false}:{loading:boolean;error:string;empty?:boolean}) {
  if (loading) return <div className="state"><span className="loader"/> Loading verified evidence…</div>;
  if (error) return <div className="state error"><strong>Request unavailable</strong><span>{error}</span></div>;
  if (empty) return <div className="state">No evidence matches the selected filters.</div>;
  return null;
}
export function MetricStrip({metrics}:{metrics:Metrics}) {
  return <div className="metric-strip">
    <div><span>MSE</span><strong>{metrics.mse.toFixed(4)}</strong></div>
    <div><span>MAE</span><strong>{metrics.mae.toFixed(4)}</strong></div>
    <div><span>PSNR</span><strong>{metrics.psnr?.toFixed(2) ?? "∞"} dB</strong></div>
    <div><span>SSIM</span><strong>{metrics.ssim.toFixed(3)}</strong></div>
  </div>;
}
export function ImageFrame({src,label,accent=false}:{src:string;label:string;accent?:boolean}) {
  return <figure className={`image-frame ${accent ? "accent" : ""}`}><img src={src} alt={label}/><figcaption>{label}</figcaption></figure>;
}
