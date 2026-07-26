import { useEffect, useState, type ReactNode } from "react";
import { api } from "../api/client";
import { Link, useRouter } from "../routes/Router";

const navigation = [
  ["/", "Overview", "OV"],
  ["/regression", "Regression", "RG"],
  ["/classification", "Classification", "CL"],
  ["/batch", "Batch studio", "BT"],
  ["/experiments", "Experiments", "EX"],
  ["/about", "About", "AB"],
];

export function AppShell({ children }: { children: ReactNode }) {
  const { path: activePath } = useRouter();
  const [ready, setReady] = useState<"checking" | "ready" | "degraded">("checking");

  useEffect(() => {
    api
      .health()
      .then((health) => setReady(health.status === "ready" ? "ready" : "degraded"))
      .catch(() => setReady("degraded"));
  }, []);

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            PT
          </div>
          <div>
            <strong>PyTorch</strong>
            <span>Tabular Studio</span>
          </div>
        </div>
        <p className="nav-label">Workspace</p>
        <nav aria-label="Primary navigation">
          {navigation.map(([path, label, icon]) => (
            <Link
              key={path}
              to={path}
              className={activePath === path ? "nav-link active" : "nav-link"}
            >
              <span className="nav-icon" aria-hidden="true">
                {icon}
              </span>
              {label}
            </Link>
          ))}
        </nav>
        <div className={`engine-status ${ready}`}>
          <span className="status-dot" />
          <div>
            <strong>{ready === "ready" ? "Both models ready" : "Engine status"}</strong>
            <span>{ready === "checking" ? "Checking bundles" : ready}</span>
          </div>
        </div>
      </aside>
      <div className="content-frame">
        <header className="topbar">
          <div>
            <span>Deep Learning Core</span>
            <strong>Project 08 · v1.0.0</strong>
          </div>
          <div className="topbar-badge">
            <span className="status-dot" /> CPU inference
          </div>
        </header>
        <main id="main-content">{children}</main>
      </div>
    </div>
  );
}
