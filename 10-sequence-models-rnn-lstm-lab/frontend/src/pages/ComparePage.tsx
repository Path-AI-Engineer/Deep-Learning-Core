import { api } from "../api/client";
import { MiniLineChart } from "../components/Charts";
import { ErrorPanel, LoadingPanel, PageHeader, StatusBanner, formatPercent } from "../components/Primitives";
import { useResource } from "../hooks/useResource";

export function ComparePage() {
  const resource = useResource(api.comparison);
  return (
    <div className="page">
      <PageHeader eyebrow="Fair comparison protocol" title="Different memory, same evidence." description="RNN, LSTM and GRU share the same fixture split, preprocessing, seed, optimization budget and validation selection rule." />
      {resource.loading && <LoadingPanel />}
      {resource.error && <ErrorPanel message={resource.error} onRetry={resource.reload} />}
      {resource.data && (
        <>
          <StatusBanner kind="warning">{resource.data.warning}</StatusBanner>
          <section className="comparison-summary">
            <div><p className="eyebrow">Approved in fixture protocol</p><h2>{resource.data.approved_model.toUpperCase()}</h2><p>Selected by validation macro F1. A tie may be broken by simplicity, parameter cost and stability.</p></div>
            <div className="comparison-scale"><span>Selection metric</span><strong>{resource.data.selection_metric.replaceAll("_", " ")}</strong><small>Test does not choose the architecture.</small></div>
          </section>
          <section className="panel table-panel">
            <div className="panel-heading"><div><p className="eyebrow">Measured results</p><h2>Model comparison</h2></div></div>
            <div className="table-scroll"><table><thead><tr><th>Model</th><th>Uses order</th><th>Parameters</th><th>Accuracy</th><th>Macro F1</th><th>Training</th></tr></thead><tbody>
              <tr><td><strong>Majority class</strong><small>Trivial baseline</small></td><td>No</td><td>0</td><td>—</td><td>—</td><td>Instant</td></tr>
              <tr><td><strong>Statistics MLP</strong><small>Order-invariant baseline</small></td><td>No</td><td>Not bundled</td><td>—</td><td>—</td><td>Offline</td></tr>
              {resource.data.models.map((model) => <tr key={model.model_id} className={model.model_id === resource.data!.approved_model ? "approved-row" : ""}><td><strong>{model.model_id.toUpperCase()}</strong><small>{model.model_id === resource.data!.approved_model ? "Approved fixture model" : "Candidate"}</small></td><td>Yes</td><td>{model.parameters.toLocaleString()}</td><td>{formatPercent(model.accuracy)}</td><td>{formatPercent(model.macro_f1)}</td><td>{model.training_seconds.toFixed(2)} s</td></tr>)}
            </tbody></table></div>
          </section>
          <section className="section-grid two">
            <article className="panel"><div className="panel-heading"><div><p className="eyebrow">Quality profile</p><h2>Accuracy vs macro F1</h2></div></div><MiniLineChart series={[{ name: "Accuracy", values: resource.data.models.map((model) => model.accuracy) }, { name: "Macro F1", values: resource.data.models.map((model) => model.macro_f1) }]} labels={resource.data.models.map((model) => model.model_id.toUpperCase())} /></article>
            <article className="panel"><div className="panel-heading"><div><p className="eyebrow">Temporal ablation contract</p><h2>Does order matter?</h2></div></div><div className="ablation-card"><span>Original sequence</span><div className="sequence-bars">{Array.from({ length: 24 }, (_, index) => <i key={index} style={{ height: `${25 + (index % 6) * 10}%` }} />)}</div><span>Permuted timesteps</span><div className="sequence-bars shuffled">{Array.from({ length: 24 }, (_, index) => <i key={index} style={{ height: `${25 + ((index * 7) % 6) * 10}%` }} />)}</div><p>The official UCI run evaluates the same labels after a seeded timestep permutation. No result is pre-claimed in fixture mode.</p></div></article>
          </section>
          <div className="tradeoff-grid">{[["RNN", "Smallest recurrent core", "Most exposed to unstable long gradient paths."], ["LSTM", "Explicit additive cell state", "Highest parameter count in this comparison."], ["GRU", "Compact gated memory", "No separate cell state to inspect."]].map(([name, strength, limit]) => <article key={name}><span>{name}</span><strong>{strength}</strong><p>{limit}</p></article>)}</div>
        </>
      )}
    </div>
  );
}
