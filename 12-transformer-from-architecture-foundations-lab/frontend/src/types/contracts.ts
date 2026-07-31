export type TaskId = "copy" | "reverse" | "recall";
export type TraceType = "encoder_self" | "decoder_self" | "cross";

export interface Health {
  status: "ready" | "degraded";
  api_version: string;
  model_registry_status: string;
  active_model: string | null;
  experiment_registry_status: string;
  detail?: string | null;
}

export interface Sample {
  example_id: string;
  task: TaskId;
  split: string;
  seed: number;
  source_tokens: string[];
  target_tokens: string[];
  content_length: number;
  canonical_hash: string;
}

export interface Prediction {
  request_id: string;
  task: TaskId;
  normalized_source: string[];
  prediction: string[];
  target: string[];
  exact_match: number;
  token_accuracy: number;
  eos_correct: number;
  eos_status: string;
  decoding_steps: Array<{
    step: number;
    selected_token_id: number;
    top_k: Array<{token_id: number; probability: number}>;
  }>;
  latency_ms: number;
  model_version: string;
  length_regime: string;
  warning?: string | null;
}

export interface Trace {
  schema_version: string;
  task: TaskId;
  trace_type: TraceType;
  layer: number;
  head: number;
  query_tokens: string[];
  key_tokens: string[];
  weights: number[][];
  shape: number[];
  entropy: number[];
  warning: string;
}

export interface EvaluationSummary {
  models: string[];
  id: {count: number; exact_match: number; token_accuracy: number};
  ood: {count: number; exact_match: number; token_accuracy: number};
  generalization_gap: number;
  per_task: Record<string, Record<string, {count: number; exact_match: number; token_accuracy: number}>>;
  cost: {
    latency: {median_ms: number; p90_ms: number};
    training: {best_epoch: number; training_seconds: number; optimizer_steps: number};
  };
  selected_bundle: string;
  decision: string;
}

