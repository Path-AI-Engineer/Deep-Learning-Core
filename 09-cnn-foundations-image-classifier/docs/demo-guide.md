# Product demo guide

1. Open **Overview** and use the tensor stack to explain how spatial dimensions
   contract while channel depth grows.
2. Open **Convolution lab**, choose an edge kernel and run the operation. Point
   out the explicit output matrix and PyTorch parity result.
3. Open **Classify**, choose an official held-out gallery image and run
   inference. Show the preprocessed input, top probabilities, model version,
   runtime and request ID.
4. Upload one PNG or JPEG and explain the visible out-of-domain warning.
5. Open **Feature maps**, select a whitelisted layer and capture its channels.
   State that activations are not causal explanations.
6. Open **Evaluation** and compare accuracy, macro F1, per-class F1, confusion
   matrix and the fair MLP baseline.
7. Close on **About the model** to show intended use, architecture and
   limitations bundled with the model.

If the header says **Degraded mode**, do not present classification metrics or
predictions. Prepare the official data and approved bundle first.
