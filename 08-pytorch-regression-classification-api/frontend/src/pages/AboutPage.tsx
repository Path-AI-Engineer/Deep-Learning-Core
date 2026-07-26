export function AboutPage() {
  return (
    <div className="page">
      <section className="hero compact">
        <div>
          <p className="eyebrow">ABOUT THE SYSTEM</p>
          <h1>Small networks. Complete lifecycle.</h1>
          <p className="hero-copy">
            Project 08 is an applied bridge from neural-network mechanics to a reproducible,
            testable inference product.
          </p>
        </div>
      </section>
      <div className="about-grid">
        <article className="panel">
          <p className="eyebrow">ARCHITECTURE</p>
          <h2>Three explicit boundaries</h2>
          <ol className="architecture-list">
            <li><span>01</span><div><strong>PyTorch engine</strong><p>Data, models, training, evaluation and bundles.</p></div></li>
            <li><span>02</span><div><strong>FastAPI contract</strong><p>Versioned inference with controlled validation.</p></div></li>
            <li><span>03</span><div><strong>React studio</strong><p>Schema-led interaction and visible model evidence.</p></div></li>
          </ol>
        </article>
        <article className="panel">
          <p className="eyebrow">WHAT IT DOES NOT CLAIM</p>
          <h2>Deliberately bounded</h2>
          <ul className="limitation-list large">
            <li>No online training, AutoML or uploaded checkpoints.</li>
            <li>No database, authentication or hidden persistence.</li>
            <li>No causal, financial or quality-certification conclusions.</li>
            <li>No certainty from classification probabilities.</li>
            <li>No selection or tuning against the isolated test set.</li>
          </ul>
        </article>
      </div>
    </div>
  );
}
