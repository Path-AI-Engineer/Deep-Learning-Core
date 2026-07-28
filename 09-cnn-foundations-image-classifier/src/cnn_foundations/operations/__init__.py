"""Educational convolution and shape operations."""

from cnn_foundations.operations.manual_convolution import cross_correlate_2d
from cnn_foundations.operations.receptive_field import receptive_field_trace
from cnn_foundations.operations.shapes import conv_output_size

__all__ = ["conv_output_size", "cross_correlate_2d", "receptive_field_trace"]

