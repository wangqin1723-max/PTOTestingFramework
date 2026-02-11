import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from pto_test.core import environment
from pto_test.core.test_case import DataType, PTOTestCase, TensorSpec

# Add pypto to path
_PYPTO_PYTHON = environment.get_pypto_python_path()
if _PYPTO_PYTHON is not None and _PYPTO_PYTHON.exists() and str(_PYPTO_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYPTO_PYTHON))


class TestMatmul(PTOTestCase):
    def __init__(self, rows: int = 64, cols: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols

    def get_name(self) -> str:
        return f"matmul_{self.rows}x{self.cols}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.rows, self.cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self.rows, self.cols], DataType.FP32, init_value=3.0),
            TensorSpec("c", [self.rows, self.cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class MatmulProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def matmul(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a_l1 = pl.op.block.load(a, [0, 0], [64, 64], target_memory=2)
                tile_b_l1 = pl.op.block.load(b, [0, 0], [64, 64], target_memory=2)
                tile_a_l0a = pl.op.block.move(tile_a_l1, target_memory=3)
                tile_b_l0b = pl.op.block.move(tile_b_l1, target_memory=4)
                tile_c_l0c = pl.op.block.matmul(tile_a_l0a, tile_b_l0b)
                # store can support l0c -> GM directly
                out_c = pl.op.block.l0c_store(tile_c_l0c, [0, 0], [64, 64], c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self, a: pl.Tensor[[64, 64], pl.FP32], b: pl.Tensor[[64, 64], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.matmul(a, b)
                return out_c

        return MatmulProgram

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = np.matmul(tensors["a"], tensors["b"])


class TestMatmulOperations:
    """Test suite for elementwise operations."""

    @pytest.mark.parametrize("rows,cols", [(64, 64)])
    def test_matmul_shapes(self, test_runner, rows, cols):
        """Test tile addition with various shapes."""
        test_case = TestMatmul(rows=rows, cols=cols)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for {rows}x{cols}: {result.error}"
