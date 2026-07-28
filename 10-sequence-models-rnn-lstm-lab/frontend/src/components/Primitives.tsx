import type { ReactNode } from "react";

export function StatusBanner({
  kind,
  children
}: {
  kind: "info" | "warning" | "error" | "success";
  children: ReactNode;
}) {
  return (
    <div className={`status-banner status-${kind}`} role={kind === "error" ? "alert" : "status"}>
      <span className="status-dot" />
      <div>{children}</div>
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  action
}: {
  eyebrow: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="page-lead">{description}</p>
      </div>
      {action}
    </header>
  );
}

export function Metric({
  label,
  value,
  detail,
  accent = "cyan"
}: {
  label: string;
  value: string;
  detail: string;
  accent?: "cyan" | "violet" | "amber";
}) {
  return (
    <article className={`metric metric-${accent}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

export function LoadingPanel({ label = "Loading laboratory evidence" }: { label?: string }) {
  return (
    <div className="loading-panel" role="status">
      <span className="loading-line" />
      <span>{label}</span>
    </div>
  );
}

export function ErrorPanel({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="error-panel" role="alert">
      <strong>Evidence could not be loaded</strong>
      <p>{message}</p>
      {onRetry && (
        <button className="button button-secondary" onClick={onRetry}>
          Retry request
        </button>
      )}
    </div>
  );
}

export function EmptyPanel({ title, message }: { title: string; message: string }) {
  return (
    <div className="empty-panel">
      <strong>{title}</strong>
      <p>{message}</p>
    </div>
  );
}

export function formatPercent(value: number | null | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}
