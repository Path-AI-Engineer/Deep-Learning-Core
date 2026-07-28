import { PageHeader, StatusBanner } from "../components/Primitives";

export function AboutPage() {
  return (
    <div className="page">
      <PageHeader eyebrow="Architecture and limitations" title="A lab with explicit boundaries." description="Sequence Memory Lab turns recurrent equations into a reproducible CPU application without pretending internal states are causal explanations." />
      <section className="architecture-map">
        {["UCI HAR ZIP", "Nine inertial channels", "[N, 128, 9]", "Train-only normalization", "RNN · LSTM · GRU", "State dict bundle", "FastAPI", "React lab"].map((item, index) => <div key={item}><span>{String(index + 1).padStart(2, "0")}</span><strong>{item}</strong>{index < 7 && <i />}</div>)}
      </section>
      <section className="section-grid two">
        <article className="panel prose-panel"><p className="eyebrow">Sequence contract</p><h2>Order is part of the input</h2><code>input: [batch, time, features] = [N, 128, 9]</code><p><strong>batch_first=True</strong> changes the input and output layout, but hidden and cell states remain organized by layers, directions and batch.</p><p>The task is many-to-one: the final recurrent representation feeds one six-class logit vector per window.</p></article>
        <article className="panel prose-panel"><p className="eyebrow">Training contract</p><h2>Selection without test leakage</h2><code>backward → measure norm → clip → optimizer.step</code><p>Validation macro F1 selects a checkpoint. The official test set stays isolated until the architecture and preprocessing are frozen.</p><p>CrossEntropyLoss receives raw logits; Softmax is reserved for inference presentation.</p></article>
      </section>
      <section className="equation-grid">
        <article><span>RNN</span><code>hₜ = tanh(Wᵢₕxₜ + bᵢₕ + Wₕₕhₜ₋₁ + bₕₕ)</code><p>Direct recurrent memory with shared parameters.</p></article>
        <article><span>LSTM</span><code>cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ</code><p>An additive cell path regulated by four gates.</p></article>
        <article><span>GRU</span><code>hₜ = (1 − zₜ) ⊙ nₜ + zₜ ⊙ hₜ₋₁</code><p>A compact gated state without separate cell memory.</p></article>
      </section>
      <section className="panel"><div className="panel-heading"><div><p className="eyebrow">Data provenance</p><h2>Human Activity Recognition Using Smartphones</h2></div><a className="text-link" href="https://doi.org/10.24432/C54S4K" target="_blank" rel="noreferrer">DOI 10.24432/C54S4K</a></div><div className="fact-grid"><div><span>Subjects</span><strong>30</strong></div><div><span>Sampling</span><strong>50 Hz</strong></div><div><span>Window</span><strong>2.56 s</strong></div><div><span>License</span><strong>CC BY 4.0</strong></div></div></section>
      <StatusBanner kind="warning"><strong>Known limits:</strong> no uploads, no live sensors, no retraining, no database, no attention and no Transformer. Those boundaries keep the project focused on recurrent memory.</StatusBanner>
    </div>
  );
}
