"""
Tests for elementwise operations using PyPTO frontend.

Tests tile-level binary operations like add, sub, mul, div.
These tests use the simplified pattern where orchestration is auto-generated.
"""

import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from pto_test.core.test_case import DataType, PTOTestCase, TensorSpec

# Add pypto to path
_FRAMEWORK_ROOT = Path(__file__).parent.parent.parent
_PYPTO_ROOT = _FRAMEWORK_ROOT / "3rdparty" / "pypto" / "python"
if _PYPTO_ROOT.exists() and str(_PYPTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYPTO_ROOT))


class TestTileAdd(PTOTestCase):
    """Test case for tile element-wise addition.

    This test case demonstrates the simplified pattern:
    - No get_orchestration() override needed (auto-generated)
    - Just implement get_program() and compute_expected()

    Note: PyPTO requires shape dimensions to be compile-time constants in type
    annotations. The shape is fixed at 128x128 for this test case.
    """

    ROWS = 128
    COLS = 128

    def __init__(self, rows: int = 128, cols: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols

    def get_name(self) -> str:
        return f"tile_add_{self.rows}x{self.cols}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            TensorSpec("a", [self.rows, self.cols], DataType.FP32, init_value=2.0),
            TensorSpec("b", [self.rows, self.cols], DataType.FP32, init_value=3.0),
            TensorSpec("c", [self.rows, self.cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        # PyPTO parser requires constant shape dimensions in type annotations.
        # Use literal values throughout.

        @pl.program
        class TileAddProgram:
            @pl.function
            def tile_add(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ):
                tile_a = pl.op.block.load(a, 0, 0, 128, 128)
                tile_b = pl.op.block.load(b, 0, 0, 128, 128)
                tile_c = pl.op.block.add(tile_a, tile_b)
                pl.op.block.store(tile_c, 0, 0, 128, 128, c)

        return TileAddProgram

    # NOTE: get_orchestration() is NOT overridden - auto-generated!

    def compute_expected(self, tensors, params=None):
        tensors["c"][:] = tensors["a"] + tensors["b"]


class TestTileMul(PTOTestCase):
    """Test case for tile element-wise multiplication."""

    def __init__(self, rows: int = 128, cols: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols

    def get_name(self) -> str:
        return f"tile_mul_{self.rows}x{self.cols}"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            # 方式1: 使用 Callable 生成随机数据（每次运行不同）
            TensorSpec("a", [self.rows, self.cols], DataType.FP32,
                      init_value=lambda shape: np.random.randn(*shape)),
            # 方式2: 使用标量值（推荐 - 简单且可序列化）
            TensorSpec("b", [self.rows, self.cols], DataType.FP32,
                      init_value=3.0),
            # 其他方式见 TestCustomArrayInit 类的示例：
            # - 小数组可以直接用 np.array([[...]])
            # - 单位矩阵用 np.eye(n)
            # - 对角矩阵用 np.diag([...])
            # 输出张量: 自动零初始化
            TensorSpec("c", [self.rows, self.cols], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        import pypto.language as pl

        @pl.program
        class TileMulProgram:
            @pl.function
            def tile_mul(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ):
                tile_a = pl.op.block.load(a, 0, 0, 128, 128)
                tile_b = pl.op.block.load(b, 0, 0, 128, 128)
                tile_c = pl.op.block.mul(tile_a, tile_b)
                pl.op.block.store(tile_c, 0, 0, 128, 128, c)

        return TileMulProgram

    # NOTE: get_orchestration() is NOT overridden - auto-generated!

    def compute_expected(self, tensors, params=None):
        # 多步计算
        temp1 = np.exp(tensors["a"])
        temp2 = np.log(tensors["b"] + 1e-8)
        temp3 = np.maximum(temp1 * temp2, 0)

        # 使用各种 NumPy 函数
        result = np.sqrt(temp3 + tensors["a"]**2)  # 注意：这里改用tensors["a"]因为只有a,b,c三个tensor
        result = np.clip(result, -100, 100)

        # 条件逻辑
        mask = tensors["a"] > 0
        result = np.where(mask, result, -result)

        tensors["c"][:] = result


class TestTileAddWithPTOAS(TestTileAdd):
    """Test tile add with PTOAS optimization strategy.

    This demonstrates how to use a custom optimization strategy.
    """

    def get_strategy(self):
        from pypto.ir.pass_manager import OptimizationStrategy
        return OptimizationStrategy.PTOAS

    def get_name(self) -> str:
        return f"tile_add_ptoas_{self.rows}x{self.cols}"


class TestCustomArrayInit(PTOTestCase):
    """Test case demonstrating custom array initialization patterns."""

    def get_name(self) -> str:
        return "custom_array_init"

    def define_tensors(self) -> List[TensorSpec]:
        return [
            # 小数组: 自定义值（会被序列化）
            TensorSpec("small", [3, 3], DataType.FP32,
                      init_value=np.array([[1, 2, 3],
                                          [4, 5, 6],
                                          [7, 8, 9]], dtype=np.float32)),
            # 单位矩阵
            TensorSpec("identity", [4, 4], DataType.FP32,
                      init_value=np.eye(4, dtype=np.float32)),
            # 常数数组（会被优化为 np.full）
            TensorSpec("constant", [5, 5], DataType.FP32,
                      init_value=np.ones((5, 5)) * 3.14),
            # 对角矩阵（小数组会序列化）
            TensorSpec("diagonal", [3, 3], DataType.FP32,
                      init_value=np.diag([1, 2, 3]).astype(np.float32)),
            # 输出
            TensorSpec("out", [3, 3], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        # Placeholder - 这个测试只是为了演示数组初始化
        return None

    def compute_expected(self, tensors, params=None):
        # 简单示例: 将 small 数组复制到输出
        tensors["out"][:] = tensors["small"][:3, :3]


# =============================================================================
# pytest test functions
# =============================================================================

class TestElementwiseOperations:
    """Test suite for elementwise operations."""

    @pytest.mark.parametrize("rows,cols", [(64, 64), (128, 128)])
    def test_tile_add_shapes(self, test_runner, rows, cols):
        """Test tile addition with various shapes."""
        test_case = TestTileAdd(rows=rows, cols=cols)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for {rows}x{cols}: {result.error}"

    @pytest.mark.parametrize("rows,cols", [(64, 64), (128, 128)])
    def test_tile_mul_shapes(self, test_runner, rows, cols):
        """Test tile multiplication with various shapes."""
        test_case = TestTileMul(rows=rows, cols=cols)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed for {rows}x{cols}: {result.error}"

    def test_tile_add_ptoas_strategy(self, test_runner):
        """Test tile addition with PTOAS optimization strategy."""
        test_case = TestTileAddWithPTOAS(rows=128, cols=128)
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"
