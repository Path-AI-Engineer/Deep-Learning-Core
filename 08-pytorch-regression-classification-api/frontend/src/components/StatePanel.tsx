export function StatePanel({
  kind,
  title,
  description,
}: {
  kind: "loading" | "error" | "empty";
  title: string;
  description: string;
}) {
  return (
    <section className={`state-panel ${kind}`} role={kind === "error" ? "alert" : "status"}>
      <span className="state-symbol" aria-hidden="true">
        {kind === "loading" ? "···" : kind === "error" ? "!" : "—"}
      </span>
      <div>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
    </section>
  );
}
