import type { ReactNode } from "react";
import {
  ActivityIcon,
  CompareIcon,
  GridIcon,
  InfoIcon,
  MatrixIcon,
  MemoryIcon,
  PulseIcon
} from "./Icons";

const navigation = [
  { path: "/", label: "Overview", detail: "System map", icon: GridIcon },
  { path: "/classify", label: "Classify", detail: "Activity signals", icon: ActivityIcon },
  { path: "/sequence-lab", label: "Sequence lab", detail: "States and gates", icon: MemoryIcon },
  { path: "/compare", label: "Compare", detail: "Model evidence", icon: CompareIcon },
  { path: "/evaluation", label: "Evaluation", detail: "Metrics and errors", icon: MatrixIcon },
  { path: "/about", label: "About", detail: "Contracts and limits", icon: InfoIcon }
];

export function Shell({
  path,
  navigate,
  children
}: {
  path: string;
  navigate: (path: string) => void;
  children: ReactNode;
}) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <button className="brand" onClick={() => navigate("/")} aria-label="Go to overview">
          <span className="brand-mark"><PulseIcon /></span>
          <span><strong>Sequence</strong><small>Memory Lab</small></span>
        </button>
        <p className="nav-label">Laboratory</p>
        <nav aria-label="Primary navigation">
          {navigation.map(({ path: target, label, detail, icon: Icon }) => (
            <a
              href={target}
              key={target}
              className={path === target ? "nav-link active" : "nav-link"}
              aria-current={path === target ? "page" : undefined}
              onClick={(event) => {
                event.preventDefault();
                navigate(target);
              }}
            >
              <Icon />
              <span><strong>{label}</strong><small>{detail}</small></span>
            </a>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span className="live-dot" />
          <div><strong>CPU inference</strong><small>Bounded local runtime</small></div>
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div><span>Deep Learning Core</span><strong>{navigation.find((item) => item.path === path)?.label}</strong></div>
          <div className="topbar-status"><span className="live-dot" /> Project 10 · v1.0.0</div>
        </header>
        <main>{children}</main>
      </div>
    </div>
  );
}
