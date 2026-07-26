import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type {
  ModelCard,
  PredictionResponse,
  TaskName,
  TaskSchema,
} from "../types/contracts";
import { LossChart } from "./LossChart";
import { MetricCard } from "./MetricCard";
import { StatePanel } from "./StatePanel";

export function TaskStudio({ task }: { task: TaskName }) {
  const [schema, setSchema] = useState<TaskSchema | null>(null);
  const [card, setCard] = useState<ModelCard | null>(null);
  const [values, setValues] = useState<Record<string, number>>({});
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  useEffect(() => {
    Promise.all([api.schema(task), api.modelCard(task)])
      .then(([nextSchema, nextCard]) => {
        setSchema(nextSchema);
        setCard(nextCard);
        setValues(nextSchema.examples[0] ?? {});
      })
      .catch((reason: Error) => setError(reason.message));
  }, [task]);

  const title = task === "regression" ? "Estimate district value." : "Classify a wine profile.";
  const eyebrow = task === "regression" ? "REGRESSION STUDIO" : "CLASSIFICATION STUDIO";
  const metricEntries = useMemo(
    () =>
      card
        ? Object.entries(card.metrics).filter(
            ([, value]) => typeof value === "number",
          ) as [string, number][]
        : [],
    [card],
  );

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setPending(true);
    setError("");
    try {
      setPrediction(await api.predict(task, values));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Prediction failed.");
    } finally {
      setPending(false);
    }
  }

  if (error && !schema) {
    return (
      <div className="page">
        <StatePanel kind="error" title="This model is unavailable" description={error} />
      </div>
    );
  }
  if (!schema || !card) {
    return (
      <div className="page">
        <StatePanel
          kind="loading"
          title="Loading the approved model"
          description="Validating schema, metrics and bundle evidence."
        />
      </div>
    );
  }

  return (
    <div className="page">
      <section className="hero compact">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h1>{title}</h1>
          <p className="hero-copy">
            Use the active {schema.dataset} model, inspect its evidence and keep every result
            inside the model’s documented boundaries.
          </p>
        </div>
        <div className="model-chip">
          <span className="status-dot" />
          <div>
            <strong>Model {card.model_version}</strong>
            <span>{schema.features.length} ordered features</span>
          </div>
        </div>
      </section>

      <div className="studio-grid">
        <form className="studio-form panel" onSubmit={submit}>
          <div className="section-heading">
            <div>
              <p className="eyebrow">INPUT WORKBENCH</p>
              <h2>Observed feature values</h2>
            </div>
            <button
              type="button"
              className="text-button"
              onClick={() => setValues(schema.examples[0])}
            >
              Load example
            </button>
          </div>
          <div className="feature-grid">
            {schema.features.map((feature) => (
              <label key={feature.name}>
                <span>{feature.display_name}</span>
                <small>{feature.description}</small>
                <input
                  type="number"
                  step="any"
                  min={feature.minimum}
                  max={feature.maximum}
                  value={values[feature.name] ?? ""}
                  onChange={(event) =>
                    setValues((current) => ({
                      ...current,
                      [feature.name]: Number(event.target.value),
                    }))
                  }
                  required
                />
                <em>
                  Range {feature.minimum.toFixed(2)} – {feature.maximum.toFixed(2)}
                </em>
              </label>
            ))}
          </div>
          {error && <p className="inline-error">{error}</p>}
          <button className="primary-button" disabled={pending}>
            {pending ? "Running inference…" : "Run approved model"}
            <span aria-hidden="true">→</span>
          </button>
        </form>

        <aside className="prediction-panel panel" aria-live="polite">
          <p className="eyebrow">MODEL OUTPUT</p>
          {!prediction ? (
            <StatePanel
              kind="empty"
              title="Ready for one observation"
              description="Load the reviewed example or provide values within the observed ranges."
            />
          ) : task === "regression" ? (
            <>
              <span className="result-label">Estimated median value</span>
              <strong className="result-value">
                ${((prediction.prediction.value ?? 0) * 100_000).toLocaleString(undefined, {
                  maximumFractionDigits: 0,
                })}
              </strong>
              <p className="result-unit">
                {prediction.prediction.value?.toFixed(3)} × {prediction.prediction.unit}
              </p>
            </>
          ) : (
            <>
              <span className="result-label">Predicted class</span>
              <strong className="result-value class-name">
                {prediction.prediction.class_name}
              </strong>
              <div className="probability-list">
                {Object.entries(prediction.prediction.probabilities ?? {}).map(
                  ([name, probability]) => (
                    <div key={name}>
                      <span>
                        {name} <b>{(probability * 100).toFixed(1)}%</b>
                      </span>
                      <div className="probability-track">
                        <i style={{ width: `${probability * 100}%` }} />
                      </div>
                    </div>
                  ),
                )}
              </div>
              <p className="certainty-note">Probability is a model output, not certainty.</p>
            </>
          )}
          {prediction?.warnings.map((warning) => (
            <p className="warning-note" key={warning}>
              {warning}
            </p>
          ))}
        </aside>
      </div>

      <section className="evidence-section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">TEST EVIDENCE</p>
            <h2>Performance with context</h2>
          </div>
          <p>Selected by validation loss · evaluated once on test</p>
        </div>
        <div className="metric-grid">
          {metricEntries.slice(0, 3).map(([name, value], index) => (
            <MetricCard
              key={name}
              label={name.replace("_", " ")}
              value={value.toFixed(3)}
              caption={
                typeof card.baseline_metrics[name] === "number"
                  ? `Baseline ${Number(card.baseline_metrics[name]).toFixed(3)}`
                  : "Approved test result"
              }
              accent={index === 0 ? "orange" : "cyan"}
            />
          ))}
        </div>
        <div className="panel chart-panel">
          <div>
            <p className="eyebrow">TRAINING TRACE</p>
            <h2>Loss across epochs</h2>
          </div>
          <LossChart
            train={card.history.train_loss}
            validation={card.history.validation_loss}
          />
        </div>
      </section>
    </div>
  );
}
