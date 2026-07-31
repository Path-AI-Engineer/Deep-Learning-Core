import type {ReactNode} from "react";
import type {Health} from "../types/contracts";
import {Icon} from "./Icons";
import {Status} from "./Primitives";

export const NAVIGATION = [
  {path: "/", label: "Overview", icon: "overview"},
  {path: "/attention-math", label: "Attention Math", icon: "sigma"},
  {path: "/masks-positions", label: "Masks & Positions", icon: "mask"},
  {path: "/architecture-trace", label: "Architecture Trace", icon: "trace"},
  {path: "/transduction", label: "Sequence Transduction", icon: "sequence"},
  {path: "/attention-explorer", label: "Attention Explorer", icon: "attention"},
  {path: "/experiments", label: "Experiments", icon: "experiment"},
  {path: "/paper", label: "Paper & Limits", icon: "paper"},
] as const;

export function Shell({children, active, navigate, health, mobileOpen, setMobileOpen}: {children: ReactNode; active: string; navigate: (path: string) => void; health: Health | null; mobileOpen: boolean; setMobileOpen: (value: boolean) => void}) {
  return <div className="app-shell">
    <aside className={mobileOpen ? "sidebar open" : "sidebar"}>
      <div className="brand"><span className="brand-mark"><Icon name="layers" /></span><div><strong>Transformer Lab</strong><small>Architecture instrument</small></div></div>
      <p className="nav-label">Research workspace</p>
      <nav aria-label="Primary navigation">{NAVIGATION.map(item => <button className={active === item.path ? "nav-item active" : "nav-item"} onClick={() => {navigate(item.path); setMobileOpen(false);}} key={item.path}><Icon name={item.icon} /><span>{item.label}</span></button>)}</nav>
      <div className="sidebar-status">
        <Status tone={health?.status === "ready" ? "ready" : "warning"}>{health?.status === "ready" ? "Reference ready" : "Setup required"}</Status>
        <small>{health?.active_model ?? "No bundle loaded"}</small>
      </div>
    </aside>
    {mobileOpen && <button className="scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
    <div className="workspace">
      <header className="topbar"><button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Icon name="menu" /></button><div><span>AI Engineer · Project 12</span><strong>{NAVIGATION.find(item => item.path === active)?.label ?? "Overview"}</strong></div><div className="top-meta"><Status tone={health?.status === "ready" ? "ready" : "warning"}>{health?.status === "ready" ? "Engine online" : "Degraded"}</Status><span className="version">v1.0</span></div></header>
      <main id="main-content">{children}</main>
    </div>
  </div>;
}

