# Model Card

## Intended use

Sequence Memory Lab demonstrates auditable six-class activity classification and recurrent-model inspection. It is educational portfolio software, not a safety, health, employment, or surveillance system.

## Current release

The committed RNN, LSTM, and GRU bundles are trained on a deterministic HAR-shaped fixture. Their metrics verify end-to-end behavior only. They must not be reported as UCI HAR performance.

## Inputs and outputs

Input is 128 ordered readings across nine inertial channels. Output is a six-class probability distribution, predicted activity, model version, evidence status, and optional hidden-state trace.

## Limitations

Fixture signals are synthetic and simplified. The models do not establish identity, intent, health, or real-world safety. Distribution shift, sensor placement, sampling differences, and subject characteristics can invalidate predictions.

