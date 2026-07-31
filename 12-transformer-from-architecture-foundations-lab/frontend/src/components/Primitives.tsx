import type {ReactNode} from "react";

export function PageHeader({eyebrow, title, copy, meta}: {eyebrow: string; title: string; copy: string; meta?: ReactNode}) {
  return (
    <header className="page-hero">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="hero-copy">{copy}</p>
      </div>
      {meta && <div className="hero-meta">{meta}</div>}
    </header>
  );
}

export function Panel({children, className = ""}: {children: ReactNode; className?: string}) {
  return <section className={`panel ${className}`}>{children}</section>;
}

export function PanelHeader({label, title, action}: {label: string; title: string; action?: ReactNode}) {
  return <div className="panel-header"><div><p className="eyebrow">{label}</p><h2>{title}</h2></div>{action}</div>;
}

export function Metric({label, value, detail, tone = "default"}: {label: string; value: string; detail: string; tone?: string}) {
  return <div className={`metric ${tone}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

export function Status({children, tone = "ready"}: {children: ReactNode; tone?: "ready" | "warning" | "error" | "neutral"}) {
  return <span className={`status ${tone}`}><i />{children}</span>;
}

export function State({kind, title, copy}: {kind: "loading" | "error" | "empty"; title: string; copy: string}) {
  return <div className={`state ${kind}`} role={kind === "error" ? "alert" : "status"}><span className="state-orbit" /><strong>{title}</strong><p>{copy}</p></div>;
}

export function TokenRail({tokens, tone = "source"}: {tokens: string[]; tone?: "source" | "target" | "prediction"}) {
  return <div className={`token-rail ${tone}`} aria-label={`${tone} tokens`}>{tokens.map((token, index) => <span key={`${token}-${index}`}>{token.replace("SYMBOL_", "S")}</span>)}</div>;
}

export function Bar({value, label, tone = "violet"}: {value: number; label: string; tone?: "violet" | "cyan" | "amber"}) {
  return <div className="bar-row"><div><span>{label}</span><strong>{(value * 100).toFixed(1)}%</strong></div><div className="bar-track"><i className={tone} style={{width: `${Math.max(2, value * 100)}%`}} /></div></div>;
}

