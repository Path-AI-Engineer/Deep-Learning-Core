# BPTT and Gradient Flow

Backpropagation through time unfolds a recurrent computation and applies the chain rule across all timesteps. Repeated derivatives can shrink toward zero or grow rapidly.

The gradient lab evaluates delayed dependencies at multiple lengths under vanishing, stable, and growing recurrent scales. Gradient clipping caps excessive norms after `backward()` and before the optimizer step. It protects numerical stability but cannot restore information already lost to vanishing gradients.

Run `python scripts/run_cell_labs.py` to produce reproducible JSON evidence.

