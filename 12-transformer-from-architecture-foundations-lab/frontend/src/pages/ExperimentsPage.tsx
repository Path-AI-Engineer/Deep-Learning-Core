import {api} from "../api/client";
import {Bar, Metric, PageHeader, Panel, PanelHeader, State, Status} from "../components/Primitives";
import {useResource} from "../hooks/useResource";

export function ExperimentsPage() {
  const resource = useResource(() => Promise.all([api.evaluation(), api.byLength(), api.errors()]));
  if (resource.loading) return <State kind="loading" title="Reading experiment registry" copy="Loading only recorded validation evidence." />;
  if (resource.error || !resource.data) return <State kind="error" title="Experiment evidence unavailable" copy={resource.error ?? "No evidence returned."} />;
  const [summary, lengths, errors] = resource.data;
  const rows = Object.entries(lengths.validation_ood ?? {}) as Array<[string, {count: number; exact_match: number; token_accuracy: number}]>;
  return <>
    <PageHeader eyebrow="Experiment registry · validation only" title="Compare evidence without crossing the frozen-test boundary." copy="The reference bundle records its seed, configuration, ID/OOD split and cost. A single run is useful for integration evidence, not statistical certainty." meta={<Status tone="warning">1 seed · test unopened</Status>} />
    <div className="metric-grid four">
      <Metric label="ID exact match" value={`${(summary.id.exact_match * 100).toFixed(1)}%`} detail={`${summary.id.count} evaluated examples`} tone="violet" />
      <Metric label="OOD exact match" value={`${(summary.ood.exact_match * 100).toFixed(1)}%`} detail={`${summary.ood.count} longer examples`} tone="cyan" />
      <Metric label="Median latency" value={`${summary.cost.latency.median_ms.toFixed(1)} ms`} detail="CPU · greedy · batch 1" />
      <Metric label="Training time" value={`${summary.cost.training.training_seconds.toFixed(1)} s`} detail={`${summary.cost.training.optimizer_steps} optimizer steps`} tone="amber" />
    </div>
    <div className="two-column">
      <Panel>
        <PanelHeader label="Length extrapolation" title="Validation OOD by content length" />
        {rows.length ? rows.map(([length, metric]) => <Bar key={length} label={`Length ${length} · n=${metric.count}`} value={metric.exact_match} tone="cyan" />) : <State kind="empty" title="No length rows" copy="The bundle does not expose OOD length evidence." />}
      </Panel>
      <Panel>
        <PanelHeader label="Selection decision" title="What was approved" />
        <div className="registry-card"><span>Selected bundle</span><strong>{summary.selected_bundle}</strong><p>{summary.decision}</p></div>
        <div className="registry-card"><span>Observed failures</span><strong>{errors.count}</strong><p>Only stored prediction mismatches are counted; absence is not proof of universal correctness.</p></div>
      </Panel>
    </div>
    <Panel>
      <PanelHeader label="Task evidence" title="ID and OOD remain disaggregated" />
      <div className="evidence-table" role="table" aria-label="Per-task validation evidence">
        <div role="row"><strong>Task</strong><strong>ID exact</strong><strong>OOD exact</strong><strong>ID token</strong><strong>OOD token</strong></div>
        {["copy", "reverse", "recall"].map(task => {
          const id = summary.per_task.validation_id?.[task];
          const ood = summary.per_task.validation_ood?.[task];
          return <div role="row" key={task}><span>{task}</span><span>{id ? `${(id.exact_match * 100).toFixed(1)}%` : "—"}</span><span>{ood ? `${(ood.exact_match * 100).toFixed(1)}%` : "—"}</span><span>{id ? `${(id.token_accuracy * 100).toFixed(1)}%` : "—"}</span><span>{ood ? `${(ood.token_accuracy * 100).toFixed(1)}%` : "—"}</span></div>;
        })}
      </div>
    </Panel>
  </>;
}
