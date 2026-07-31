import {api} from "../api/client";
import {PageHeader, Panel, PanelHeader, State, Status} from "../components/Primitives";
import {useResource} from "../hooks/useResource";

export function PaperPage() {
  const resource = useResource(() => Promise.all([api.modelCard(), api.research()]));
  if (resource.loading) return <State kind="loading" title="Loading model card" copy="Reading the protocol and registered limits." />;
  if (resource.error || !resource.data) return <State kind="error" title="Research record unavailable" copy={resource.error ?? "No record returned."} />;
  const [card, research] = resource.data;
  const limits = card.limitations as string[];
  const questions = research.research_questions as string[];
  const threats = research.threats as string[];
  return <>
    <PageHeader eyebrow="Reproducibility record" title="The paper starts where the demo stops." copy="Purpose, protocol, evidence status and threats travel with the model bundle. The interface never upgrades validation observations into final research claims." meta={<Status tone="warning">{String(research.results_status).replace("_", " ")}</Status>} />
    <div className="paper-grid">
      <Panel><PanelHeader label="Research questions" title="What the lab is allowed to ask" /><ol className="research-list">{questions.map((question, index) => <li key={question}><span>{String(index + 1).padStart(2, "0")}</span><p>{question}</p></li>)}</ol></Panel>
      <Panel><PanelHeader label="Evidence protocol" title="Current boundary" /><div className="protocol-card"><strong>{String(research.protocol)}</strong><p>Hypotheses: {String(research.hypotheses_status).replaceAll("_", " ")}.</p></div><div className="paper-meta"><span>Model version<strong>{String(card.version)}</strong></span><span>Tasks<strong>{(card.tasks as string[]).join(" · ")}</strong></span></div></Panel>
    </div>
    <div className="two-column">
      <Panel><PanelHeader label="Model limitations" title="Claims we refuse to make" /><ul className="limit-list">{limits.map(limit => <li key={limit}>{limit}</li>)}</ul></Panel>
      <Panel><PanelHeader label="Threats to validity" title="What could mislead interpretation" /><ul className="limit-list threats">{threats.map(threat => <li key={threat}>{threat}</li>)}</ul></Panel>
    </div>
    <Panel className="repro-panel"><PanelHeader label="Artifact chain" title="Reproduce from data contract to figure" /><div className="repro-flow">{["Deterministic suite", "Versioned configuration", "Hash-verified model", "Recorded metrics", "Paper assets"].map((item, index) => <div key={item}><span>{String(index + 1).padStart(2, "0")}</span><strong>{item}</strong></div>)}</div></Panel>
  </>;
}
