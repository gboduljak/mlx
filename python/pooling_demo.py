import mlx.core as mx
from mlx.nn.layers import MaxPooling2d

a = mx.arange(16).reshape((4, 4))
pool_no_padding = MaxPooling2d(kernel_size=2, stride=2, padding=0)
print(pool_no_padding(a))
print("--")
pool_with_padding = MaxPooling2d(kernel_size=2, stride=2, padding=1)
print(pool_with_padding(a))
