import type { ReactNode } from "react";
import { Icon } from "./Icons";

export type RouteId =
  | "overview"
  | "classify"
  | "convolution"
  | "feature-maps"
  | "evaluation"
  | "about";

const routes: Array<{ id: RouteId; label: string; path: string; icon: string }> = [
  { id: "overview", label: "Overview", path: "/", icon: "overview" },
  { id: "classify", label: "Classify", path: "/classify", icon: "classify" },
  { id: "convolution", label: "Convolution lab", path: "/convolution", icon: "convolution" },
  { id: "feature-maps", label: "Feature maps", path: "/feature-maps", icon: "features" },
  { id: "evaluation", label: "Evaluation", path: "/evaluation", icon: "evaluation" },
  { id: "about", label: "About the model", path: "/about", icon: "about" }
];

export function Shell({
  active,
  onNavigate,
  children,
  ready,
  mobileOpen,
  setMobileOpen
}: {
  active: RouteId;
  onNavigate: (path: string) => void;
  children: ReactNode;
  ready: boolean;
  mobileOpen: boolean;
  setMobileOpen: (open: boolean) => void;
}) {
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <aside className={mobileOpen ? "sidebar sidebar--open" : "sidebar"}>
        <div className="brand">
          <div className="brand-mark" aria-hidden="true"><Icon name="spark" size={22}/></div>
          <div><strong>CNN Vision Lab</strong><span>Spatial intelligence</span></div>
        </div>
        <button className="mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close navigation">
          <Icon name="close"/>
        </button>
        <p className="nav-label">Workspace</p>
        <nav aria-label="Primary navigation">
          {routes.map((route) => (
            <button
              aria-current={route.id === active ? "page" : undefined}
              className={route.id === active ? "nav-item nav-item--active" : "nav-item"}
              key={route.id}
              onClick={() => { onNavigate(route.path); setMobileOpen(false); }}
            >
              <Icon name={route.icon}/><span>{route.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span className={ready ? "status-dot" : "status-dot status-dot--warn"}/>
          <div><strong>{ready ? "Model ready" : "Setup required"}</strong><span>FashionMNIST · v1</span></div>
        </div>
      </aside>
      {mobileOpen && <button className="scrim" onClick={() => setMobileOpen(false)} aria-label="Close navigation"/>}
      <section className="workspace">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileOpen(true)} aria-label="Open navigation">
            <Icon name="menu"/>
          </button>
          <div><span>AI ENGINEER · PROJECT 09</span><strong>{routes.find((route) => route.id === active)?.label}</strong></div>
          <div className="runtime-pill"><span className={ready ? "status-dot" : "status-dot status-dot--warn"}/>{ready ? "Inference ready" : "Degraded mode"}</div>
        </header>
        <main id="main-content">{children}</main>
      </section>
    </div>
  );
}
