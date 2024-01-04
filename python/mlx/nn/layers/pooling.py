# Copyright © 2023 Apple Inc.


import mlx.core as mx
from mlx.nn.layers.base import Module


class MaxPooling2d(Module):
    def __init__(self, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, a):
        # For the sake of simplicity, assumes that a is two dimensional.
        a = mx.pad(a, self.padding)
        # Compute sliding windows by creating a strided view
        output_shape = (
            (a.shape[0] - self.kernel_size) // self.stride + 1,
            (a.shape[1] - self.kernel_size) // self.stride + 1,
        )
        windows_shape = (
            output_shape[0],
            output_shape[1],
            self.kernel_size,
            self.kernel_size,
        )
        windows_strides = (
            self.stride * a.strides[0],
            self.stride * a.strides[1],
            a.strides[0],
            a.strides[1],
        )
        windows = mx.as_strided(a, windows_shape, windows_strides)
        # Set reduction axes
        reduction_axes = (2, 3)
        # For demo purposes
        print(a)
        print(windows)
        # Reduce over all windows
        return mx.max(windows, reduction_axes)
