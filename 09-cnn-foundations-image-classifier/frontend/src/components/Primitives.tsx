import type { ReactNode } from "react";

export function PageHeader({
  eyebrow,
  title,
  description,
  aside
}: {
  eyebrow: string;
  title: string;
  description: string;
  aside?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p>{description}</p></div>
      {aside}
    </header>
  );
}

export function Panel({
  title,
  eyebrow,
  children,
  className = ""
}: {
  title: string;
  eyebrow?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <header>{eyebrow && <p className="panel-eyebrow">{eyebrow}</p>}<h2>{title}</h2></header>
      {children}
    </section>
  );
}

export function EmptyState({
  title,
  children
}: {
  title: string;
  children: ReactNode;
}) {
  return <div className="empty-state"><span className="empty-orbit"/><h3>{title}</h3><p>{children}</p></div>;
}

export function ErrorBanner({ message }: { message: string }) {
  return <div className="error-banner" role="alert"><strong>Request unavailable</strong><span>{message}</span></div>;
}

export function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}
