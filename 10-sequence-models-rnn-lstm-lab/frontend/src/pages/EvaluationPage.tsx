import { useState } from "react";
import { api } from "../api/client";
import { ConfusionMatrix } from "../components/Charts";
import { ErrorPanel, LoadingPanel, Metric, PageHeader, StatusBanner, formatPercent } from "../components/Primitives";
import { useResource } from "../hooks/useResource";
import type { ModelId } from "../types/contracts";

const labels = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"];

export function EvaluationPage() {
  const [modelId, setModelId] = useState<ModelId>("rnn");
  const comparison = useResource(api.comparison);
  const model = comparison.data?.models.find((item) => item.model_id === modelId);
  return (
    <div className="page">
      <PageHeader eyebrow="Evaluation and error analysis" title="Read beyond one score." description="Inspect class balance, confusion structure and the gap between overall accuracy and macro-averaged performance." action={<label className="inline-select">Model<select value={modelId} onChange={(event) => setModelId(event.target.value as ModelId)}><option value="rnn">RNN</option><option value="lstm">LSTM</option><option value="gru">GRU</option></select></label>} />
      {comparison.loading && <LoadingPanel />}
      {comparison.error && <ErrorPanel message={comparison.error} onRetry={comparison.reload} />}
      {model && <>
        <StatusBanner kind="warning">Fixture evaluation validates the pipeline and UI. It must not be published as UCI HAR performance.</StatusBanner>
        <section className="metric-grid four"><Metric label="Accuracy" value={formatPercent(model.accuracy)} detail="All fixture predictions" /><Metric label="Macro F1" value={formatPercent(model.macro_f1)} detail="Equal weight per class" accent="violet" /><Metric label="Macro precision" value={formatPercent(model.macro_precision)} detail="Across six activities" accent="amber" /><Metric label="Test loss" value={model.test_loss.toFixed(3)} detail="CrossEntropyLoss on logits" /></section>
        <section className="section-grid evaluation-grid">
          <article className="panel"><div className="panel-heading"><div><p className="eyebrow">Count matrix</p><h2>Where activities are confused</h2></div></div><ConfusionMatrix values={model.confusion_matrix} labels={labels} /></article>
          <article className="panel"><div className="panel-heading"><div><p className="eyebrow">Per-class evidence</p><h2>Six activity profiles</h2></div></div><div className="class-metrics">{model.per_class.map((metric) => <div key={metric.class_name}><span>{metric.class_name.replaceAll("_", " ")}</span><div><i style={{ width: `${metric.f1 * 100}%` }} /></div><strong>{formatPercent(metric.f1)}</strong><small>{metric.support} samples</small></div>)}</div></article>
        </section>
        <section className="panel"><div className="panel-heading"><div><p className="eyebrow">Interpretation</p><h2>Errors are part of the evidence</h2></div></div><div className="error-notes"><article><span>01</span><strong>Postural similarity</strong><p>Sitting and standing can share low dynamic energy; orientation becomes important.</p></article><article><span>02</span><strong>Walking family</strong><p>Level walking and stairs share periodic structure while impact and vertical components differ.</p></article><article><span>03</span><strong>Subject generalization</strong><p>The official protocol keeps people separated to avoid optimistic window-level leakage.</p></article></div></section>
      </>}
    </div>
  );
}
