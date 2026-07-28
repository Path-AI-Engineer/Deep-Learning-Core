import { useEffect, useState } from "react";
import { EmptyState, ErrorBanner, Metric, PageHeader, Panel } from "../components/Primitives";
import { api, type Evaluation } from "../lib/api";

export function EvaluationPage() {
  const [data, setData] = useState<Evaluation | null>(null);
  const [error, setError] = useState("");
  useEffect(() => { api.evaluation().then(setData).catch((reason: Error) => setError(reason.message)); }, []);
  return (
    <>
      <PageHeader eyebrow="HELD-OUT TEST EVIDENCE" title="Evaluate without moving the goalposts." description="The CNN and MLP baseline share the same split contract. Test metrics are generated only after validation-based model selection."/>
      {error && <ErrorBanner message={error}/>}
      {data ? <>
        <div className="metrics-grid">
          <Metric label="CNN accuracy" value={`${(data.metrics.accuracy * 100).toFixed(1)}%`} detail={`${data.metrics.test_examples.toLocaleString()} held-out samples`}/>
          <Metric label="Macro F1" value={data.metrics.macro_f1.toFixed(3)} detail="Equal weight across 10 classes"/>
          <Metric label="MLP accuracy" value={`${(data.comparison.mlp.accuracy * 100).toFixed(1)}%`} detail="Same split and test policy"/>
          <Metric label="Best epoch" value={String(data.training_history.best_epoch)} detail="Selected on validation loss"/>
        </div>
        <div className="evaluation-grid">
          <Panel eyebrow="PER-CLASS EVIDENCE" title="Where performance differs"><div className="class-table" role="table">{data.per_class_metrics.classes.map((row) => <div role="row" key={row.index}><span>{row.class_name}</span><div><i style={{ width: `${row.f1 * 100}%` }}/></div><strong>{row.f1.toFixed(3)}</strong><small>n={row.support}</small></div>)}</div></Panel>
          <Panel eyebrow="CONFUSION MATRIX" title="Predicted versus observed"><div className="confusion" style={{ gridTemplateColumns: `repeat(${data.confusion_matrix.matrix.length}, 1fr)` }}>{data.confusion_matrix.matrix.flatMap((row, y) => row.map((value, x) => { const max = Math.max(...row, 1); return <span title={`Observed ${data.confusion_matrix.labels[y]}, predicted ${data.confusion_matrix.labels[x]}: ${value}`} key={`${y}-${x}`} style={{ background: `rgba(49,216,192,${0.05 + value / max * 0.8})` }}>{value}</span>; }))}</div><p className="matrix-note">Rows are observed labels; columns are predicted labels. Hover a cell for detail.</p></Panel>
        </div>
      </> : <Panel title="Evaluation bundle"><EmptyState title="No approved metrics loaded">Build the versioned CNN bundle after training both experiments on the official data.</EmptyState></Panel>}
    </>
  );
}
