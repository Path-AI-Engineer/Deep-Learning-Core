import { useEffect, useState } from "react";
import { api } from "../api/client";
import { LossChart } from "../components/LossChart";
import { StatePanel } from "../components/StatePanel";
import type { ModelCard } from "../types/contracts";

export function ExperimentsPage() {
  const [cards, setCards] = useState<ModelCard[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([api.modelCard("regression"), api.modelCard("classification")])
      .then(setCards)
      .catch((reason: Error) => setError(reason.message));
  }, []);

  return (
    <div className="page">
      <section className="hero compact">
        <div>
          <p className="eyebrow">APPROVED EXPERIMENTS</p>
          <h1>Evidence before interface.</h1>
          <p className="hero-copy">
            Every active model comes from an immutable run with fixed splits, baseline
            comparison, validation-selected checkpoint and isolated test evaluation.
          </p>
        </div>
      </section>
      {error && <StatePanel kind="error" title="Evidence unavailable" description={error} />}
      {!error && !cards.length && (
        <StatePanel kind="loading" title="Loading model cards" description="Reading bundle evidence." />
      )}
      <div className="experiment-list">
        {cards.map((card) => (
          <article className="panel experiment-card" key={card.task}>
            <div className="experiment-header">
              <div>
                <p className="eyebrow">{card.task.toUpperCase()} · MODEL {card.model_version}</p>
                <h2>{card.dataset}</h2>
              </div>
              <span>{card.architecture.hidden_units.join(" → ")} hidden units</span>
            </div>
            <div className="experiment-content">
              <LossChart train={card.history.train_loss} validation={card.history.validation_loss} />
              <div>
                <h3>Test metrics</h3>
                <dl className="metric-list">
                  {Object.entries(card.metrics)
                    .filter(([, value]) => typeof value === "number")
                    .map(([key, value]) => (
                      <div key={key}>
                        <dt>{key.replace("_", " ")}</dt>
                        <dd>{Number(value).toFixed(4)}</dd>
                      </div>
                    ))}
                </dl>
                <h3>Boundaries</h3>
                <ul className="limitation-list">
                  {card.limitations.map((limitation) => (
                    <li key={limitation}>{limitation}</li>
                  ))}
                </ul>
              </div>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
