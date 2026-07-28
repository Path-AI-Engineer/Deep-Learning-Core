# Padding and Packing Lab

Create unequal sequence lengths, pad to a common batch shape, and pass the original lengths to the recurrent encoder. Confirm that packed execution uses the final valid timestep rather than padded zeros.

The API endpoint `/api/v1/labs/padding` exposes a reproducible worked example.
