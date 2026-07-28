export type Health = {
  status: "ready" | "degraded";
  api_version: string;
  model_available: boolean;
  gallery_available: boolean;
  model_version: string | null;
  training_mode: boolean;
};

export type Sample = {
  sample_id: string;
  label_index: number;
  class_name: string;
  image_data_url: string;
  split: string;
};

export type Prediction = {
  predicted_index: number;
  predicted_class: string;
  true_class: { index: number; name: string } | null;
  probabilities: { index: number; class_name: string; probability: number }[];
  top_k: { index: number; class_name: string; probability: number }[];
  model_version: string;
  inference_time_ms: number;
  preprocessed_preview: string;
  preprocessing_summary: string[];
  warnings: string[];
  request_id: string;
};

export type Evaluation = {
  metrics: { accuracy: number; macro_f1: number; test_examples: number };
  per_class_metrics: {
    classes: Array<{
      index: number;
      class_name: string;
      precision: number;
      recall: number;
      f1: number;
      support: number;
    }>;
  };
  confusion_matrix: { labels: string[]; matrix: number[][] };
  training_history: {
    epochs: Array<{
      epoch: number;
      train: { loss: number; accuracy: number };
      validation: { loss: number; accuracy: number };
    }>;
    best_epoch: number;
  };
  comparison: {
    cnn: { accuracy: number; macro_f1: number };
    mlp: { accuracy: number; macro_f1: number };
    protocol: string;
  };
  limitations: string[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const body = await response.json().catch(() => ({ detail: "Invalid server response." }));
  if (!response.ok) {
    throw new Error(
      typeof body.detail === "string" ? body.detail : `Request failed (${response.status}).`
    );
  }
  return body as T;
}

export const api = {
  health: () => request<Health>("/api/v1/health"),
  samples: () => request<{ items: Sample[] }>("/api/v1/samples?limit=20"),
  predictSample: (sampleId: string) =>
    request<Prediction>("/api/v1/predictions/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_id: sampleId })
    }),
  predictUpload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<Prediction>("/api/v1/predictions/upload", {
      method: "POST",
      body: form
    });
  },
  convolution: (matrix: number[][], kernel: number[][], stride: number, padding: number) =>
    request<{
      output: number[][];
      output_shape: number[];
      operation_trace: unknown[];
      parity_result: { passed: boolean; max_absolute_error: number; operation: string };
    }>("/api/v1/labs/convolution", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ matrix, kernel, stride, padding })
    }),
  activations: (sampleId: string, layerId: string) =>
    request<{
      layer_id: string;
      tensor_shape: number[];
      feature_maps: number[][][];
      predicted_index: number;
      interpretation_warning: string;
      sample: Sample;
    }>("/api/v1/explanations/activations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_id: sampleId, layer_id: layerId, limit: 8 })
    }),
  evaluation: () => request<Evaluation>("/api/v1/evaluation/summary"),
  modelCard: () =>
    request<{
      dataset: string;
      architecture: Record<string, unknown>;
      input: { shape: number[]; dtype: string };
      metrics: Record<string, number>;
      limitations: string[];
      domain: string;
    }>("/api/v1/model-card")
};
