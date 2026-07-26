export type TaskName = "regression" | "classification";

export interface TaskStatus {
  task: TaskName;
  available: boolean;
  model_version: string | null;
  dataset: string | null;
}

export interface FeatureSpec {
  name: string;
  display_name: string;
  description: string;
  minimum: number;
  maximum: number;
  example: number;
  unit: string | null;
}

export interface TaskSchema {
  task: TaskName;
  dataset: string;
  feature_names: string[];
  features: FeatureSpec[];
  class_names: string[];
  target_unit: string | null;
  examples: Record<string, number>[];
}

export interface ModelCard {
  task: TaskName;
  model_version: string;
  dataset: string;
  architecture: {
    input_features: number;
    hidden_units: number[];
    dropout: number;
    output_units: number;
  };
  metrics: Record<string, number | number[][]>;
  baseline_metrics: Record<string, number | number[][]>;
  history: { train_loss: number[]; validation_loss: number[] };
  limitations: string[];
}

export interface PredictionResponse {
  task: TaskName;
  model_version: string;
  prediction: {
    value?: number;
    unit?: string;
    class_name?: string;
    probabilities?: Record<string, number>;
  };
  warnings: string[];
}
