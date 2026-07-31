import type {ReactNode} from "react";
import {Icon} from "./Icons";

const links = [
  ["/","overview","Overview"],
  ["/reconstruct","reconstruct","Reconstruct"],
  ["/denoise","denoise","Denoise"],
  ["/latent","latent","Latent Explorer"],
  ["/interpolate","interpolate","Interpolate"],
  ["/compare","compare","Compare"],
  ["/evaluation","evaluation","Evaluation"],
  ["/about","about","About & Limits"],
];
export function Shell({children,path}:{children:ReactNode;path:string}) {
  return <div className="app-shell">
    <aside className="sidebar">
      <a href="/" className="brand" aria-label="Latent Representation Lab home"><span className="brand-mark"><span/></span><span><strong>Latent Lab</strong><small>Representation systems</small></span></a>
      <nav aria-label="Primary navigation">{links.map(([href,icon,label]) => <a key={href} href={href} className={path===href ? "active" : ""}><Icon name={icon}/><span>{label}</span></a>)}</nav>
      <div className="side-status"><span className="status-dot"/><div><strong>Evidence ready</strong><small>Offline fixture · v1.0.0</small></div></div>
    </aside>
    <div className="workspace">
      <header><div><small>DEEP LEARNING CORE · PROJECT 11</small><strong>Latent Representation Lab</strong></div><div className="runtime"><span className="status-dot"/> Local inference</div></header>
      <main>{children}</main>
    </div>
  </div>;
}
