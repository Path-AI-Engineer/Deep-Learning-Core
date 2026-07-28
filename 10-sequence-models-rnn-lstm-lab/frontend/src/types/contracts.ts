export type ModelId = "rnn" | "lstm" | "gru";

export interface Health {
  status: "ready" | "degraded";
  api_version: string;
  data_mode: string;
  bundles_available: ModelId[];
  active_model: ModelId | null;
  model_versions: Record<string, string>;
}

export interface ModelRow {
  model_id: ModelId;
  type: string;
  version: string;
  hidden_size: number;
  parameters: number;
  metrics: {
    accuracy: number | null;
    macro_f1: number | null;
    validation_macro_f1: number | null;
  };
  latency_ms: number | null;
  available: boolean;
  active: boolean;
}

export interface SampleSummary {
  sample_id: string;
  activity: string;
  subject_id: string;
  split: string;
  sequence_length: number;
  channels: number;
  preview: number[][];
}

export interface SampleDetail {
  sample_id: string;
  activity: string;
  subject_id: string;
  split: string;
  sequence_length: number;
  channels: string[];
  signals: number[][];
  data_mode: string;
}

export interface Probability {
  class_name: string;
  probability: number;
}

export interface Prediction {
  prediction_id: string;
  sample_id: string;
  true_class: string;
  predicted_class: string;
  probabilities: Probability[];
  top_k: Probability[];
  confidence: number;
  model_type: ModelId;
  model_version: string;
  latency_ms: number;
  warnings: string[];
}

export interface Comparison {
  comparison_version: string;
  data_mode: string;
  selection_metric: string;
  approved_model: ModelId;
  models: Array<{
    model_id: ModelId;
    accuracy: number;
    macro_f1: number;
    macro_precision: number;
    macro_recall: number;
    validation_macro_f1: number;
    parameters: number;
    test_loss: number;
    training_seconds: number;
    confusion_matrix: number[][];
    per_class: PerClassMetric[];
  }>;
  warning: string;
}

export interface PerClassMetric {
  class_index: number;
  class_name: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface CellStep {
  timestep: number;
  input: number[];
  previous_hidden: number[];
  hidden: number[];
  hidden_norm: number;
  previous_cell?: number[];
  cell?: number[];
  gates?: Record<string, number[]>;
}

export interface CellTrace {
  cell_type: ModelId;
  timesteps: CellStep[];
  educational_output: number[];
  pytorch_output: number[];
  max_abs_difference: number;
  parity_tolerance: number;
}

export interface GradientPoint {
  length: number;
  gradient_norm_before: number;
  gradient_norm_after: number;
}

export interface GradientFlow {
  experiment: string;
  reproducible: boolean;
  scenarios: Array<{
    scenario: string;
    recurrent_scale: number;
    clipping_threshold: number | null;
    points: GradientPoint[];
  }>;
  interpretation: string;
}
