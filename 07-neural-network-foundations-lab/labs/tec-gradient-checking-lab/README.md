# Gradient Checking Lab

The lab validates all hidden/output weights and biases against central finite
differences. To observe failure detection, temporarily alter one analytical
gradient in a local experiment; the report identifies the affected parameter,
maximum absolute error and maximum relative error.

The committed implementation and tests must never preserve that intentional
defect.
