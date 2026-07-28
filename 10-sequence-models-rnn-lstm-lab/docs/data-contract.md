# Data Contract

## Official source

The production experiment targets the UCI Human Activity Recognition Using Smartphones dataset. Each observation is a fixed 128-timestep window with nine float32 inertial channels and one of six activity labels.

## Shapes

- values: `[N, 128, 9]`, finite `float32`;
- labels: `[N]`, integer range `0..5`;
- subjects: `[N]`, positive subject identifiers.

## Split policy

The official test partition remains untouched. Validation subjects are selected only from the official training partition, and subject identifiers may not overlap between train, validation, and test. Per-channel mean and standard deviation are fitted on training values only.

## Provenance

`download_manifest.json` records source URL, download time, bytes, and SHA-256. `preparation_manifest.json` records channels, labels, subjects, sizes, split policy, and normalization values.

The committed fixture has the same shape contract but is explicitly labeled `fixture-not-uci`; it validates the software, not benchmark quality.

