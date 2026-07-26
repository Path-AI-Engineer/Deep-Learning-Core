import { useState } from "react";
import { api } from "../api/client";
import { parseCsv } from "../features/batch/csv";
import type { TaskName } from "../types/contracts";

type Row = Record<string, number>;

export function BatchPage() {
  const [task, setTask] = useState<TaskName>("classification");
  const [rows, setRows] = useState<Row[]>([]);
  const [results, setResults] = useState<Record<string, unknown>[]>([]);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  async function readFile(file?: File) {
    if (!file) return;
    try {
      const parsed = parseCsv(await file.text());
      if (parsed.length > 100) throw new Error("Batch limit is 100 rows.");
      setRows(parsed);
      setResults([]);
      setError("");
    } catch (reason) {
      setRows([]);
      setError(reason instanceof Error ? reason.message : "Could not parse CSV.");
    }
  }

  async function runBatch() {
    setPending(true);
    try {
      const response = await api.predictBatch(task, rows);
      setResults(response.predictions);
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Batch inference failed.");
    } finally {
      setPending(false);
    }
  }

  function download() {
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${task}-predictions.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return (
    <div className="page">
      <section className="hero compact">
        <div>
          <p className="eyebrow">BATCH STUDIO</p>
          <h1>Validate before you run.</h1>
          <p className="hero-copy">
            Upload a small numeric CSV, inspect its shape, execute no more than 100 observations
            and download model results without storing source data.
          </p>
        </div>
      </section>
      <div className="batch-grid">
        <section className="panel upload-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">01 · INPUT</p>
              <h2>Choose the task and CSV</h2>
            </div>
          </div>
          <div className="segmented-control">
            {(["regression", "classification"] as TaskName[]).map((name) => (
              <button
                key={name}
                className={task === name ? "selected" : ""}
                onClick={() => setTask(name)}
              >
                {name}
              </button>
            ))}
          </div>
          <label className="dropzone">
            <span>CSV</span>
            <strong>Choose a numeric dataset</strong>
            <small>Header names must match the active schema · maximum 100 rows</small>
            <input type="file" accept=".csv,text/csv" onChange={(event) => readFile(event.target.files?.[0])} />
          </label>
          {error && <p className="inline-error">{error}</p>}
          <button className="primary-button" disabled={!rows.length || pending} onClick={runBatch}>
            {pending ? "Running batch…" : `Run ${rows.length || 0} rows`}
            <span>→</span>
          </button>
        </section>
        <section className="panel preview-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">02 · PREVIEW</p>
              <h2>{results.length ? "Inference results" : "Validated observations"}</h2>
            </div>
            {results.length > 0 && (
              <button className="text-button" onClick={download}>
                Download JSON
              </button>
            )}
          </div>
          {!rows.length ? (
            <p className="empty-copy">The first five valid rows will appear here.</p>
          ) : (
            <pre>{JSON.stringify(results.length ? results.slice(0, 5) : rows.slice(0, 5), null, 2)}</pre>
          )}
        </section>
      </div>
    </div>
  );
}
