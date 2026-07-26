import type {
  ModelCard,
  PredictionResponse,
  TaskName,
  TaskSchema,
  TaskStatus,
} from "../types/contracts";

const API = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed with status ${response.status}.`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () =>
    request<{ status: string; models: Record<TaskName, boolean> }>("/health/ready"),
  tasks: () => request<{ tasks: TaskStatus[]; batch_limit: number }>("/tasks"),
  schema: (task: TaskName) => request<TaskSchema>(`/tasks/${task}/schema`),
  modelCard: (task: TaskName) => request<ModelCard>(`/tasks/${task}/model-card`),
  predict: (task: TaskName, features: Record<string, number>) =>
    request<PredictionResponse>(`/predictions/${task}`, {
      method: "POST",
      body: JSON.stringify({ features }),
    }),
  predictBatch: (task: TaskName, rows: Record<string, number>[]) =>
    request<{
      count: number;
      predictions: PredictionResponse["prediction"][];
      model_version: string;
    }>(`/predictions/${task}/batch`, {
      method: "POST",
      body: JSON.stringify({ rows }),
    }),
};
