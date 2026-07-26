import { useEffect, useState } from "react";
import { api } from "../api/client";
import { StatePanel } from "../components/StatePanel";
import { Link } from "../routes/Router";
import type { TaskStatus } from "../types/contracts";

export function OverviewPage() {
  const [tasks, setTasks] = useState<TaskStatus[] | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    api
      .tasks()
      .then((payload) => setTasks(payload.tasks))
      .catch((reason: Error) => setError(reason.message));
  }, []);

  return (
    <div className="page">
      <section className="hero">
        <div>
          <p className="eyebrow">PYTORCH TABULAR STUDIO · PROJECT 08</p>
          <h1>
            From gradients to
            <span> usable inference.</span>
          </h1>
          <p className="hero-copy">
            Two neural networks, one disciplined lifecycle. Explore leakage-safe data,
            approved test evidence and CPU inference through a production-shaped application.
          </p>
          <div className="hero-actions">
            <Link className="primary-button inline" to="/regression">
              Open regression <span>→</span>
            </Link>
            <Link className="secondary-button" to="/experiments">
              Review evidence
            </Link>
          </div>
        </div>
        <div className="hero-visual" aria-label="Training to inference lifecycle">
          <div className="visual-core">PT</div>
          {["DATA", "MLP", "TEST", "API"].map((label, index) => (
            <span key={label} style={{ "--index": index } as React.CSSProperties}>
              {label}
            </span>
          ))}
        </div>
      </section>

      <section className="system-strip" aria-label="System lifecycle">
        {["Split without leakage", "Train with autograd", "Select on validation", "Serve approved bundle"].map(
          (label, index) => (
            <div key={label}>
              <span>0{index + 1}</span>
              <strong>{label}</strong>
            </div>
          ),
        )}
      </section>

      {error && <StatePanel kind="error" title="API unavailable" description={error} />}
      {!tasks && !error && (
        <StatePanel
          kind="loading"
          title="Reading model registry"
          description="Checking both approved bundles and task contracts."
        />
      )}
      {tasks && (
        <section className="task-section">
          <div className="section-heading">
            <div>
              <p className="eyebrow">ACTIVE WORKBENCHES</p>
              <h2>One system, two learning problems.</h2>
            </div>
            <p>Each model keeps its own loss, metrics and interpretation.</p>
          </div>
          <div className="task-grid">
            {tasks.map((task) => (
              <article className="task-card" key={task.task}>
                <div className={`task-symbol ${task.task}`}>{task.task === "regression" ? "R" : "C"}</div>
                <span className={`availability ${task.available ? "available" : ""}`}>
                  <i /> {task.available ? "Bundle approved" : "Bundle required"}
                </span>
                <p className="eyebrow">{task.task.toUpperCase()}</p>
                <h3>
                  {task.task === "regression"
                    ? "California Housing value"
                    : "Wine multiclass profile"}
                </h3>
                <p>
                  {task.task === "regression"
                    ? "Estimate median district value and compare MAE against a mean baseline."
                    : "Inspect the predicted class, all probabilities and macro-F1 evidence."}
                </p>
                <dl>
                  <div>
                    <dt>Dataset</dt>
                    <dd>{task.dataset ?? "Pending"}</dd>
                  </div>
                  <div>
                    <dt>Version</dt>
                    <dd>{task.model_version ?? "—"}</dd>
                  </div>
                </dl>
                <Link to={`/${task.task}`}>
                  Enter studio <span>→</span>
                </Link>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
