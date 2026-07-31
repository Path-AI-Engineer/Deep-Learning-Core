import type {
  EvaluationSummary,
  Health,
  Prediction,
  Sample,
  TaskId,
  Trace,
  TraceType,
} from "../types/contracts";

const API_ROOT = import.meta.env.VITE_API_URL ?? "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: {"Content-Type": "application/json", ...init?.headers},
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({detail: "Request failed."}));
    throw new Error(payload.detail ?? `Request failed with status ${response.status}.`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),
  architecture: () => request<Record<string, unknown>>("/architecture"),
  modelCard: () => request<Record<string, unknown>>("/model-card"),
  tasks: () => request<Array<Record<string, unknown>>>("/tasks"),
  tokens: () => request<Record<string, unknown>>("/tokens"),
  research: () => request<Record<string, unknown>>("/research"),
  samples: (task?: TaskId) =>
    request<{items: Sample[]; count: number; total: number}>(
      `/samples${task ? `?task=${task}` : ""}`,
    ),
  predict: (sample: Sample) =>
    request<Prediction>("/predict", {
      method: "POST",
      body: JSON.stringify({
        task: sample.task,
        sample_id: sample.example_id,
        max_new_tokens: 28,
      }),
    }),
  trace: (sample: Sample, traceType: TraceType, layer: number, head: number) =>
    request<Trace>("/trace", {
      method: "POST",
      body: JSON.stringify({
        task: sample.task,
        sample_id: sample.example_id,
        max_new_tokens: 28,
        trace_type: traceType,
        layer,
        head,
      }),
    }),
  attention: (payload: {
    query: number[][];
    key: number[][];
    value: number[][];
    mask?: boolean[][];
  }) =>
    request<Record<string, unknown>>("/attention/compute", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  evaluation: () => request<EvaluationSummary>("/evaluation/summary"),
  byLength: () => request<Record<string, unknown>>("/evaluation/by-length"),
  errors: () => request<{items: Array<Record<string, unknown>>; count: number}>("/evaluation/errors"),
};

