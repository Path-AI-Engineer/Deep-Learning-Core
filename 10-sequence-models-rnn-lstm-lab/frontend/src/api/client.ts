import type {
  CellTrace,
  Comparison,
  GradientFlow,
  Health,
  ModelId,
  ModelRow,
  Prediction,
  SampleDetail,
  SampleSummary
} from "../types/contracts";

const API_ROOT = "/api/v1";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers
    }
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),
  models: async () => (await request<{ models: ModelRow[] }>("/models")).models,
  comparison: () => request<Comparison>("/evaluation/summary"),
  samples: async (activity?: string) => {
    const query = activity ? `?activity=${encodeURIComponent(activity)}&limit=24` : "?limit=24";
    return (await request<{ items: SampleSummary[] }>(`/samples${query}`)).items;
  },
  sample: (sampleId: string) => request<SampleDetail>(`/samples/${sampleId}`),
  predict: (sampleId: string, modelId: ModelId | "active") =>
    request<Prediction>("/predictions/sample", {
      method: "POST",
      body: JSON.stringify({ sample_id: sampleId, model_id: modelId })
    }),
  comparePrediction: (sampleId: string) =>
    request<{ predictions: Prediction[]; agreement: boolean }>("/predictions/compare", {
      method: "POST",
      body: JSON.stringify({ sample_id: sampleId, model_ids: ["rnn", "lstm", "gru"] })
    }),
  cellTrace: (cellType: ModelId) =>
    request<CellTrace>("/labs/cell-trace", {
      method: "POST",
      body: JSON.stringify({ cell_type: cellType, example_id: "balanced-memory" })
    }),
  gradientFlow: () => request<GradientFlow>("/labs/gradient-flow"),
  evaluation: (modelId: ModelId) =>
    request<{ metrics: Comparison["models"][number] }>(`/evaluation/${modelId}`)
};
