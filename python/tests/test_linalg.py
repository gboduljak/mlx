# Copyright © 2023 Apple Inc.

import itertools
import math
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestLinalg(mlx_tests.MLXTestCase):
    def test_norm(self):
        vector_ords = [None, 0.5, 0, 1, 2, 3, -1, float("inf"), -float("inf")]
        matrix_ords = [None, "fro", -1, 1, float("inf"), -float("inf")]

        for shape in [(3,), (2, 3), (2, 3, 3)]:
            x_mx = mx.arange(1, math.prod(shape) + 1).reshape(shape)
            x_np = np.arange(1, math.prod(shape) + 1).reshape(shape)
            # Test when at least one axis is provided
            for num_axes in range(1, len(shape)):
                if num_axes == 1:
                    ords = vector_ords
                else:
                    ords = matrix_ords
                for axis in itertools.combinations(range(len(shape)), num_axes):
                    for keepdims in [True, False]:
                        for o in ords:
                            out_np = np.linalg.norm(
                                x_np, ord=o, axis=axis, keepdims=keepdims
                            )
                            out_mx = mx.linalg.norm(
                                x_mx, ord=o, axis=axis, keepdims=keepdims
                            )
                            with self.subTest(
                                shape=shape, ord=o, axis=axis, keepdims=keepdims
                            ):
                                self.assertTrue(
                                    np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6)
                                )

        # Test only ord provided
        for shape in [(3,), (2, 3)]:
            x_mx = mx.arange(1, math.prod(shape) + 1).reshape(shape)
            x_np = np.arange(1, math.prod(shape) + 1).reshape(shape)
            for o in [None, 1, -1, float("inf"), -float("inf")]:
                for keepdims in [True, False]:
                    out_np = np.linalg.norm(x_np, ord=o, keepdims=keepdims)
                    out_mx = mx.linalg.norm(x_mx, ord=o, keepdims=keepdims)
                    with self.subTest(shape=shape, ord=o, keepdims=keepdims):
                        self.assertTrue(
                            np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6)
                        )

        # Test no ord and no axis provided
        for shape in [(3,), (2, 3), (2, 3, 3)]:
            x_mx = mx.arange(1, math.prod(shape) + 1).reshape(shape)
            x_np = np.arange(1, math.prod(shape) + 1).reshape(shape)
            for keepdims in [True, False]:
                out_np = np.linalg.norm(x_np, keepdims=keepdims)
                out_mx = mx.linalg.norm(x_mx, keepdims=keepdims)
                with self.subTest(shape=shape, keepdims=keepdims):
                    self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
