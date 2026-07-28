import { api } from "../api/client";
import { ArrowIcon, MemoryIcon, PulseIcon } from "../components/Icons";
import {
  ErrorPanel,
  LoadingPanel,
  Metric,
  PageHeader,
  StatusBanner,
  formatPercent
} from "../components/Primitives";
import { useResource } from "../hooks/useResource";

export function OverviewPage({ navigate }: { navigate: (path: string) => void }) {
  const resource = useResource(() =>
    Promise.all([api.health(), api.models(), api.comparison()]).then(
      ([health, models, comparison]) => ({ health, models, comparison })
    )
  );

  return (
    <div className="page">
      <PageHeader
        eyebrow="Sequence intelligence · Project 10"
        title="See what memory does over time."
        description="Inspect how recurrent states carry, transform and forget information across 128 sensor timesteps."
        action={
          <button className="button button-primary" onClick={() => navigate("/classify")}>
            Open classifier <ArrowIcon />
          </button>
        }
      />
      {resource.loading && <LoadingPanel />}
      {resource.error && <ErrorPanel message={resource.error} onRetry={resource.reload} />}
      {resource.data && (
        <>
          {resource.data.health.data_mode === "fixture" && (
            <StatusBanner kind="warning">
              <strong>Educational fixture mode.</strong> The entire inference path is real, while
              the bundled measurements validate software behavior—not UCI HAR performance claims.
            </StatusBanner>
          )}
          <section className="overview-hero">
            <div className="overview-copy">
              <span className="shape-chip">[N, 128, 9]</span>
              <h2>From nine inertial signals to one activity.</h2>
              <p>
                A controlled many-to-one pipeline compares a simple RNN, an LSTM and a GRU under
                the same sequence, seed and evaluation contract.
              </p>
              <div className="flow-row" aria-label="Sequence processing flow">
                {["Signals", "Normalize", "Remember", "Classify"].map((step, index) => (
                  <div key={step}>
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <strong>{step}</strong>
                  </div>
                ))}
              </div>
            </div>
            <div className="memory-orbit" aria-label="Abstract recurrent memory state visualization">
              <div className="orbit orbit-one" />
              <div className="orbit orbit-two" />
              <div className="memory-core"><MemoryIcon /></div>
              <span className="state-node node-a">xₜ</span>
              <span className="state-node node-b">hₜ</span>
              <span className="state-node node-c">cₜ</span>
              <span className="state-node node-d">ŷ</span>
            </div>
          </section>
          <section className="metric-grid four">
            <Metric label="Active model" value={(resource.data.health.active_model ?? "—").toUpperCase()} detail="Selected by validation macro F1" />
            <Metric label="Fixture test F1" value={formatPercent(resource.data.comparison.models.find((item) => item.model_id === resource.data!.health.active_model)?.macro_f1)} detail="Measured, not a UCI benchmark" accent="violet" />
            <Metric label="Input contract" value="128 × 9" detail="Timesteps × sensor channels" accent="amber" />
            <Metric label="Activities" value="6" detail="Walking, postural and stairs" />
          </section>
          <section className="section-grid two">
            <article className="panel">
              <div className="panel-heading"><div><p className="eyebrow">Architecture field notes</p><h2>Three ways to carry state</h2></div><PulseIcon /></div>
              <div className="architecture-list">
                {[
                  ["RNN", "One hidden state", "Direct recurrence exposes gradient fragility."],
                  ["LSTM", "Hidden + cell state", "Four gates regulate an additive memory path."],
                  ["GRU", "Compact hidden state", "Three gates balance memory and parameter cost."]
                ].map(([name, state, copy]) => (
                  <button key={name} onClick={() => navigate("/sequence-lab")}>
                    <span>{name}</span><div><strong>{state}</strong><small>{copy}</small></div><ArrowIcon />
                  </button>
                ))}
              </div>
            </article>
            <article className="panel">
              <div className="panel-heading"><div><p className="eyebrow">Runtime readiness</p><h2>Evidence, not decoration</h2></div><span className="live-dot" /></div>
              <ul className="evidence-list">
                <li><span>API</span><strong>{resource.data.health.status === "ready" ? "Ready" : "Degraded"}</strong></li>
                <li><span>Bundles</span><strong>{resource.data.health.bundles_available.length} / 3</strong></li>
                <li><span>Cell parity</span><strong>RNN · LSTM · GRU</strong></li>
                <li><span>Gradient lab</span><strong>Precomputed and bounded</strong></li>
              </ul>
              <button className="text-link" onClick={() => navigate("/compare")}>Review comparison evidence <ArrowIcon /></button>
            </article>
          </section>
        </>
      )}
    </div>
  );
}
