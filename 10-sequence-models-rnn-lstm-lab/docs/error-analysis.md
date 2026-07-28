# Error Analysis

Evaluation reports confusion matrix, per-class precision, recall, F1, support, and macro averages. Macro F1 is the selection metric because each activity should contribute equally.

Analysis should distinguish dynamic classes (walking and stairs) from posture classes (sitting, standing, laying). Confusion between sitting and standing may reflect similar low-motion windows; confusion among locomotion classes may reflect transition timing and subject variation.

Before accepting a model, inspect class support, train/validation divergence, permutation ablation, gradient history, and incorrect sequences. The current fixture results are too small and synthetic for real-world error claims.

