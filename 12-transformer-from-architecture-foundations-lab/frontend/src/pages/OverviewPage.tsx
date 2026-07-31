import {api} from "../api/client";
import {Icon} from "../components/Icons";
import {Bar, Metric, PageHeader, Panel, PanelHeader, State, Status} from "../components/Primitives";
import {useResource} from "../hooks/useResource";

export function OverviewPage({navigate}: {navigate: (path: string) => void}) {
  const resource = useResource(() => Promise.all([api.architecture(), api.evaluation()]));
  if (resource.loading) return <State kind="loading" title="Loading architecture evidence" copy="Reading the active bundle and validation registry." />;
  if (resource.error || !resource.data) return <State kind="error" title="Architecture evidence unavailable" copy={resource.error ?? "No response was returned."} />;
  const [architecture, evaluation] = resource.data;
  const config = architecture.active_config as Record<string, string | number>;
  return <>
    <PageHeader eyebrow="Architecture first · evidence bounded" title="See how a sequence becomes another sequence." copy="A manual encoder–decoder Transformer, three controlled tasks and one visual instrument for following the mathematics into real inference." meta={<div className="hero-badge"><span>Q</span><i>→</i><span>K</span><i>×</i><span>V</span></div>} />
    <div className="metric-grid four">
      <Metric label="Validation ID" value={`${(evaluation.id.exact_match * 100).toFixed(1)}%`} detail={`${evaluation.id.count} controlled examples`} tone="violet" />
      <Metric label="Validation OOD" value={`${(evaluation.ood.exact_match * 100).toFixed(1)}%`} detail="Unseen sequence lengths" tone="cyan" />
      <Metric label="Architecture" value={`${config.encoder_layers}E · ${config.decoder_layers}D`} detail={`${config.num_heads} heads · d${config.d_model}`} />
      <Metric label="Evidence status" value="Validation" detail="Frozen test remains unopened" tone="amber" />
    </div>
    <div className="overview-grid">
      <Panel className="architecture-panel">
        <PanelHeader label="Encoder–decoder path" title="One forward pass, exposed." action={<Status>Manual primitives</Status>} />
        <div className="architecture-flow">
          {[
            ["01", "Source tokens", "Task + discrete symbols"],
            ["02", "Encoder", "Self-attention builds memory"],
            ["03", "Decoder", "Masked attention queries memory"],
            ["04", "Vocabulary logits", "Greedy EOS-bounded output"],
          ].map(([index, title, detail]) => <div className="flow-node" key={index}><span>{index}</span><div><strong>{title}</strong><small>{detail}</small></div><Icon name="arrow" /></div>)}
        </div>
      </Panel>
      <Panel>
        <PanelHeader label="Controlled suite" title="Three tasks, distinct pressure." />
        <div className="task-stack">
          <button onClick={() => navigate("/transduction")}><span>C</span><div><strong>Copy</strong><small>Monotonic alignment and EOS</small></div><Icon name="arrow" /></button>
          <button onClick={() => navigate("/transduction")}><span>R</span><div><strong>Reverse</strong><small>Order and positional signal</small></div><Icon name="arrow" /></button>
          <button onClick={() => navigate("/transduction")}><span>A</span><div><strong>Associative recall</strong><small>Content-addressed retrieval</small></div><Icon name="arrow" /></button>
        </div>
      </Panel>
    </div>
    <div className="two-column">
      <Panel>
        <PanelHeader label="Generalization lens" title="ID and OOD remain separate." />
        <Bar label="Validation ID exact match" value={evaluation.id.exact_match} />
        <Bar label="Validation OOD exact match" value={evaluation.ood.exact_match} tone="cyan" />
        <p className="panel-note">Gap: {(evaluation.generalization_gap * 100).toFixed(1)} percentage points. This is controlled length extrapolation, not language generalization.</p>
      </Panel>
      <Panel className="decision-panel">
        <PanelHeader label="Research boundary" title="What this evidence does not claim." />
        <ul className="limit-list"><li>These tasks are synthetic algorithms.</li><li>Attention weights are descriptive, not causal explanations.</li><li>The current bundle is validation evidence, not a final test claim.</li></ul>
        <button className="secondary-button" onClick={() => navigate("/paper")}>Read protocol and limits <Icon name="arrow" /></button>
      </Panel>
    </div>
  </>;
}

