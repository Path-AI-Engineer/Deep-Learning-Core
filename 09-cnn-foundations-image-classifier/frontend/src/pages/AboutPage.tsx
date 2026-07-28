import { useEffect, useState } from "react";
import { EmptyState, ErrorBanner, PageHeader, Panel } from "../components/Primitives";
import { api } from "../lib/api";

type Card = Awaited<ReturnType<typeof api.modelCard>>;

export function AboutPage() {
  const [card, setCard] = useState<Card | null>(null);
  const [error, setError] = useState("");
  useEffect(() => { api.modelCard().then(setCard).catch((reason: Error) => setError(reason.message)); }, []);
  return (
    <>
      <PageHeader eyebrow="MODEL CARD" title="Know the boundary of the model." description="Architecture, intended use, preprocessing and limitations travel with the approved bundle—not as a separate marketing claim."/>
      {error && <ErrorBanner message={error}/>}
      {card ? <div className="about-grid">
        <Panel eyebrow="INTENDED DOMAIN" title={card.dataset}><p className="large-copy">{card.domain}</p><dl className="definition-list"><div><dt>Input</dt><dd>{card.input.shape.join(" × ")} {card.input.dtype}</dd></div><div><dt>Task</dt><dd>10-class supervised classification</dd></div><div><dt>Serving</dt><dd>CPU · evaluation mode · no retraining</dd></div></dl></Panel>
        <Panel eyebrow="LIMITATIONS" title="Use with restraint"><ul className="limitation-list">{card.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul></Panel>
        <Panel eyebrow="ARCHITECTURE CONTRACT" title="Reconstructable configuration" className="wide-panel"><pre>{JSON.stringify(card.architecture, null, 2)}</pre></Panel>
      </div> : <Panel title="Model card"><EmptyState title="Bundle not loaded">The model card becomes available only with a complete, hash-verified artifact.</EmptyState></Panel>}
    </>
  );
}
