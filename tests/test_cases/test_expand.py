"""
Tests for row_expand_div operation using PyPTO frontend.

This test demonstrates the row_expand_div operation which expands a row vector
and performs element-wise division with a matrix.
"""

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


class TestRowExpandDivBase(PTOTestCase):
    """Base test case for row_expand_div operation.

    This operation takes a matrix and a column vector, and divides each row
    of the matrix by the corresponding scalar value from the column vector.

    For example:
    - Matrix a: [[6, 8], [12, 16]]
    - Column vector b: [[2], [4]]
    - Result c: [[6/2, 8/2], [12/4, 16/4]] = [[3, 4], [3, 4]]

    Note: PyPTO requires shape dimensions to be compile-time constants in type
    annotations, so each shape needs its own subclass with get_program() method.
    """

    # Subclasses must define these
    ROWS = 128
    COLS = 128

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rows = self.ROWS
        self.cols = self.COLS

    def get_name(self) -> str:
        return f"row_expand_div_{self.rows}x{self.cols}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            # Matrix to be divided (random values)
            TensorSpec("a", [self.rows, self.cols], DataType.FP32,
                      init_value=lambda shape: np.random.rand(*shape).astype(np.float32)),
            # Column vector (divisor) - shape is [rows, 1] (random values, avoid division by zero)
            TensorSpec("b", [self.rows, 1], DataType.FP32,
                      init_value=lambda shape: (np.random.rand(*shape) + 0.1).astype(np.float32)),
            # Output tensor
            TensorSpec("c", [self.rows, self.cols], DataType.FP32, is_output=True),
        ]

    def compute_expected(self, tensors, params=None):
        """Compute expected output: each row of a divided by corresponding scalar in b."""
        # Broadcasting: a[rows, cols] / b[rows, 1] -> c[rows, cols]
        tensors["c"][:] = tensors["a"] / tensors["b"]


# Generate test classes for different shapes
class TestRowExpandDiv_32x32(TestRowExpandDivBase):
    ROWS = 32
    COLS = 32

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class RowExpandDivProgram:
            @pl.function
            def row_expand_div(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                b: pl.Tensor[[1, 32], pl.FP32],
                c: pl.Tensor[[32, 32], pl.FP32],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[32, 32])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[1, 32])

                tile_b_reshaped = pl.reshape(tile_b, [32, 1])

                tile_c = pl.row_expand_div(tile_a, tile_b_reshaped)

                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[32, 32], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                b: pl.Tensor[[1, 32], pl.FP32]
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                out_c = self.row_expand_div(a, b)
                return out_c

        return RowExpandDivProgram


class TestRowExpandDiv_64x64(TestRowExpandDivBase):
    ROWS = 64
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class RowExpandDivProgram:
            @pl.function
            def row_expand_div(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 1], pl.FP32],
                c: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[64, 64])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[64, 1])
                tile_c = pl.row_expand_div(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[64, 64], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 1], pl.FP32]
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out_c = self.row_expand_div(a, b)
                return out_c

        return RowExpandDivProgram


class TestRowExpandDiv_128x128(TestRowExpandDivBase):
    ROWS = 128
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class RowExpandDivProgram:
            @pl.function
            def row_expand_div(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 1], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 1])
                tile_c = pl.row_expand_div(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                out_c = self.row_expand_div(a, b)
                return out_c

        return RowExpandDivProgram


class TestRowExpandDiv_128x64(TestRowExpandDivBase):
    ROWS = 128
    COLS = 64

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class RowExpandDivProgram:
            @pl.function
            def row_expand_div(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[128, 1], pl.FP32],
                c: pl.Tensor[[128, 64], pl.FP32],
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 64])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 1])
                tile_c = pl.row_expand_div(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[128, 64], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[128, 64], pl.FP32],
                b: pl.Tensor[[128, 1], pl.FP32]
            ) -> pl.Tensor[[128, 64], pl.FP32]:
                out_c = self.row_expand_div(a, b)
                return out_c

        return RowExpandDivProgram


class TestRowExpandDiv_64x128(TestRowExpandDivBase):
    ROWS = 64
    COLS = 128

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class RowExpandDivProgram:
            @pl.function
            def row_expand_div(
                self,
                a: pl.Tensor[[64, 128], pl.FP32],
                b: pl.Tensor[[64, 1], pl.FP32],
                c: pl.Tensor[[64, 128], pl.FP32],
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[64, 128])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[64, 1])
                tile_c = pl.row_expand_div(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[64, 128], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[64, 128], pl.FP32],
                b: pl.Tensor[[64, 1], pl.FP32]
            ) -> pl.Tensor[[64, 128], pl.FP32]:
                out_c = self.row_expand_div(a, b)
                return out_c

        return RowExpandDivProgram


class TestRowExpandDiv_96x96(TestRowExpandDivBase):
    ROWS = 96
    COLS = 96

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class RowExpandDivProgram:
            @pl.function
            def row_expand_div(
                self,
                a: pl.Tensor[[96, 96], pl.FP32],
                b: pl.Tensor[[96, 1], pl.FP32],
                c: pl.Tensor[[96, 96], pl.FP32],
            ) -> pl.Tensor[[96, 96], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[96, 96])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[96, 1])
                tile_c = pl.row_expand_div(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[96, 96], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[96, 96], pl.FP32],
                b: pl.Tensor[[96, 1], pl.FP32]
            ) -> pl.Tensor[[96, 96], pl.FP32]:
                out_c = self.row_expand_div(a, b)
                return out_c

        return RowExpandDivProgram


class TestRowExpandDiv_80x96(TestRowExpandDivBase):
    ROWS = 80
    COLS = 96

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class RowExpandDivProgram:
            @pl.function
            def row_expand_div(
                self,
                a: pl.Tensor[[80, 96], pl.FP32],
                b: pl.Tensor[[80, 1], pl.FP32],
                c: pl.Tensor[[80, 96], pl.FP32],
            ) -> pl.Tensor[[80, 96], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[80, 96])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[80, 1])
                tile_c = pl.row_expand_div(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[80, 96], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[80, 96], pl.FP32],
                b: pl.Tensor[[80, 1], pl.FP32]
            ) -> pl.Tensor[[80, 96], pl.FP32]:
                out_c = self.row_expand_div(a, b)
                return out_c

        return RowExpandDivProgram


class TestRowExpandDiv_96x80(TestRowExpandDivBase):
    ROWS = 96
    COLS = 80

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class RowExpandDivProgram:
            @pl.function
            def row_expand_div(
                self,
                a: pl.Tensor[[96, 80], pl.FP32],
                b: pl.Tensor[[96, 1], pl.FP32],
                c: pl.Tensor[[96, 80], pl.FP32],
            ) -> pl.Tensor[[96, 80], pl.FP32]:
                tile_a = pl.load(a, offsets=[0, 0], shapes=[96, 80])
                tile_b = pl.load(b, offsets=[0, 0], shapes=[96, 1])
                tile_c = pl.row_expand_div(tile_a, tile_b)
                out_c = pl.store(tile_c, offsets=[0, 0], shapes=[96, 80], output_tensor=c)
                return out_c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                a: pl.Tensor[[96, 80], pl.FP32],
                b: pl.Tensor[[96, 1], pl.FP32]
            ) -> pl.Tensor[[96, 80], pl.FP32]:
                out_c = self.row_expand_div(a, b)
                return out_c

        return RowExpandDivProgram


# =============================================================================
# pytest test functions
# =============================================================================


def test_row_expand_div_32x32(test_runner):
    """Test 32x32 shape."""
    test_case = TestRowExpandDiv_32x32()
    result = test_runner.run(test_case)
    assert result.passed, f"Test failed: {result.error}"


def test_row_expand_div_64x64(test_runner):
    """Test 64x64 shape."""
    test_case = TestRowExpandDiv_64x64()
    result = test_runner.run(test_case)
    assert result.passed, f"Test failed: {result.error}"


def test_row_expand_div_128x128(test_runner):
    """Test 128x128 shape."""
    test_case = TestRowExpandDiv_128x128()
    result = test_runner.run(test_case)
    assert result.passed, f"Test failed: {result.error}"


def test_row_expand_div_128x64(test_runner):
    """Test 128x64 shape."""
    test_case = TestRowExpandDiv_128x64()
    result = test_runner.run(test_case)
    assert result.passed, f"Test failed: {result.error}"


def test_row_expand_div_64x128(test_runner):
    """Test 64x128 shape."""
    test_case = TestRowExpandDiv_64x128()
    result = test_runner.run(test_case)
    assert result.passed, f"Test failed: {result.error}"


def test_row_expand_div_96x96(test_runner):
    """Test 96x96 shape."""
    test_case = TestRowExpandDiv_96x96()
    result = test_runner.run(test_case)
    assert result.passed, f"Test failed: {result.error}"


def test_row_expand_div_80x96(test_runner):
    """Test 80x96 shape."""
    test_case = TestRowExpandDiv_80x96()
    result = test_runner.run(test_case)
    assert result.passed, f"Test failed: {result.error}"


def test_row_expand_div_96x80(test_runner):
    """Test 96x80 shape."""
    test_case = TestRowExpandDiv_96x80()
    result = test_runner.run(test_case)
    assert result.passed, f"Test failed: {result.error}"
