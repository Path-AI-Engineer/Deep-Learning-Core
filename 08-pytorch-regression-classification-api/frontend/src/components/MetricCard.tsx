export function MetricCard({
  label,
  value,
  caption,
  accent = "orange",
}: {
  label: string;
  value: string;
  caption: string;
  accent?: "orange" | "cyan" | "neutral";
}) {
  return (
    <article className={`metric-card ${accent}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{caption}</p>
    </article>
  );
}
