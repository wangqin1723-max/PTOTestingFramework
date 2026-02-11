"""
Orchestration 组合函数生成器

该模块负责生成 @pl.function(type=pl.FunctionType.Orchestration) 函数，
用于组合多个 InCore 内核。支持三种组合模式：
- Sequential: 顺序执行内核
- Branching: 分支执行内核
- Mixed: 混合模式
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from .fuzzer import is_shape_aligned


class OrchestratorGenerator:
    """生成 Orchestration 组合函数的生成器"""

    def __init__(self, seed: Optional[int] = None):
        """初始化组合函数生成器

        Args:
            seed: 随机种子，用于可重现性
        """
        self.rng = random.Random(seed)

    def generate_sequential(
        self,
        kernels: List[Dict[str, Any]],
        shape: Tuple[int, int] = (128, 128),
    ) -> Dict[str, Any]:
        """生成顺序执行模式的 Orchestration 函数

        在顺序模式中，每个内核的输出作为下一个内核的输入。

        Args:
            kernels: 内核信息列表
            shape: 张量形状

        Returns:
            包含组合函数信息的字典
        """
        if not kernels:
            raise ValueError("至少需要一个内核")

        # 收集所有需要的输入及其形状
        input_shapes_map = {}  # {input_name: shape}
        for kernel in kernels:
            for inp_name, inp_shape in kernel["inputs"]:
                if inp_name not in input_shapes_map:
                    input_shapes_map[inp_name] = inp_shape
                # 如果同一个输入在不同内核中有不同形状，使用较大的形状
                elif inp_shape != input_shapes_map[inp_name]:
                    existing_size = input_shapes_map[inp_name][0] * input_shapes_map[inp_name][1]
                    new_size = inp_shape[0] * inp_shape[1]
                    if new_size > existing_size:
                        input_shapes_map[inp_name] = inp_shape

        # 生成函数签名
        input_params = sorted(input_shapes_map.keys())
        params = []
        for name in input_params:
            inp_shape = input_shapes_map[name]
            params.append(f"{name}: pl.Tensor[[{inp_shape[0]}, {inp_shape[1]}], pl.FP32]")

        # 输出形状使用最后一个内核的输出形状
        output_shape = kernels[-1]["output_shape"]
        rows, cols = output_shape

        code_lines = [
            "    @pl.function(type=pl.FunctionType.Orchestration)",
            f"    def orchestrator(self, {', '.join(params)}) -> pl.Tensor[[{rows}, {cols}], pl.FP32]:",
        ]

        # 顺序调用内核 - 不需要显式创建 tensor
        result_var = None
        for i, kernel in enumerate(kernels):
            kernel_name = kernel["name"]
            kernel_inputs = [inp[0] for inp in kernel["inputs"]]

            # 如果不是第一个内核，使用前一个内核的输出
            if i > 0 and result_var:
                # 替换第一个输入为前一个内核的输出
                kernel_inputs[0] = result_var

            # 调用 InCore 函数，框架会自动处理输出 tensor
            result_var = f"result_{i}"
            inputs_str = ", ".join(kernel_inputs)
            code_lines.append(f"        {result_var} = self.{kernel_name}({inputs_str})")

        # 返回最后一个结果
        code_lines.append(f"        return {result_var}")

        return {
            "mode": "sequential",
            "code": "\n".join(code_lines),
            "inputs": input_params,
            "output_shape": output_shape,
        }

    def generate_branching(
        self,
        kernels: List[Dict[str, Any]],
        shape: Tuple[int, int] = (128, 128),
    ) -> Dict[str, Any]:
        """生成分支执行模式的 Orchestration 函数

        在分支模式中，多个内核并行执行，然后合并结果。

        Args:
            kernels: 内核信息列表
            shape: 张量形状

        Returns:
            包含组合函数信息的字典
        """
        if not kernels:
            raise ValueError("至少需要一个内核")

        # 收集所有需要的输入及其形状
        input_shapes_map = {}  # {input_name: shape}
        for kernel in kernels:
            for inp_name, inp_shape in kernel["inputs"]:
                if inp_name not in input_shapes_map:
                    input_shapes_map[inp_name] = inp_shape
                # 如果同一个输入在不同内核中有不同形状，使用较大的形状
                elif inp_shape != input_shapes_map[inp_name]:
                    existing_size = input_shapes_map[inp_name][0] * input_shapes_map[inp_name][1]
                    new_size = inp_shape[0] * inp_shape[1]
                    if new_size > existing_size:
                        input_shapes_map[inp_name] = inp_shape

        # 生成函数签名
        input_params = sorted(input_shapes_map.keys())
        params = []
        for name in input_params:
            inp_shape = input_shapes_map[name]
            params.append(f"{name}: pl.Tensor[[{inp_shape[0]}, {inp_shape[1]}], pl.FP32]")

        # 输出形状：在分支模式中，所有分支应该有相同的输出形状
        # 使用第一个内核的输出形状
        output_shape = kernels[0]["output_shape"]
        rows, cols = output_shape

        code_lines = [
            "    @pl.function(type=pl.FunctionType.Orchestration)",
            f"    def orchestrator(self, {', '.join(params)}) -> pl.Tensor[[{rows}, {cols}], pl.FP32]:",
        ]

        # 并行执行所有内核 - 不需要显式创建 tensor
        result_vars = []
        for i, kernel in enumerate(kernels):
            kernel_name = kernel["name"]
            kernel_inputs = [inp[0] for inp in kernel["inputs"]]
            result_var = f"branch_{i}"
            result_vars.append(result_var)

            inputs_str = ", ".join(kernel_inputs)
            code_lines.append(f"        {result_var} = self.{kernel_name}({inputs_str})")

        # 合并所有分支结果
        if len(result_vars) == 1:
            code_lines.append(f"        return {result_vars[0]}")
        else:
            # 使用 add 操作合并结果
            code_lines.append(f"        # 合并所有分支结果")
            merged = result_vars[0]
            for i in range(1, len(result_vars)):
                new_merged = f"merged_{i}"
                code_lines.append(f"        {new_merged} = self.merge_results({merged}, {result_vars[i]})")
                merged = new_merged
            code_lines.append(f"        return {merged}")

        return {
            "mode": "branching",
            "code": "\n".join(code_lines),
            "inputs": input_params,
            "output_shape": output_shape,
            "needs_merge_kernel": len(result_vars) > 1,
        }

    def generate_mixed(
        self,
        kernels: List[Dict[str, Any]],
        shape: Tuple[int, int] = (128, 128),
    ) -> Dict[str, Any]:
        """生成混合模式的 Orchestration 函数

        混合模式结合了顺序和分支执行。

        Args:
            kernels: 内核信息列表
            shape: 张量形状

        Returns:
            包含组合函数信息的字典
        """
        if len(kernels) < 2:
            # 如果内核数量少于2，使用顺序模式
            return self.generate_sequential(kernels, shape)

        # 收集所有需要的输入及其形状
        input_shapes_map = {}  # {input_name: shape}
        for kernel in kernels:
            for inp_name, inp_shape in kernel["inputs"]:
                if inp_name not in input_shapes_map:
                    input_shapes_map[inp_name] = inp_shape
                # 如果同一个输入在不同内核中有不同形状，使用较大的形状
                elif inp_shape != input_shapes_map[inp_name]:
                    existing_size = input_shapes_map[inp_name][0] * input_shapes_map[inp_name][1]
                    new_size = inp_shape[0] * inp_shape[1]
                    if new_size > existing_size:
                        input_shapes_map[inp_name] = inp_shape

        # 生成函数签名
        input_params = sorted(input_shapes_map.keys())
        params = []
        for name in input_params:
            inp_shape = input_shapes_map[name]
            params.append(f"{name}: pl.Tensor[[{inp_shape[0]}, {inp_shape[1]}], pl.FP32]")

        # 输出形状使用最后一个内核的输出形状
        output_shape = kernels[-1]["output_shape"]
        rows, cols = output_shape

        code_lines = [
            "    @pl.function(type=pl.FunctionType.Orchestration)",
            f"    def orchestrator(self, {', '.join(params)}) -> pl.Tensor[[{rows}, {cols}], pl.FP32]:",
        ]

        # 将内核分成两组：前半部分并行，后半部分顺序
        mid = len(kernels) // 2
        parallel_kernels = kernels[:mid]
        sequential_kernels = kernels[mid:]

        # 并行执行前半部分 - 不需要显式创建 tensor
        branch_results = []
        for i, kernel in enumerate(parallel_kernels):
            kernel_name = kernel["name"]
            kernel_inputs = [inp[0] for inp in kernel["inputs"]]
            result_var = f"parallel_{i}"
            branch_results.append(result_var)

            inputs_str = ", ".join(kernel_inputs)
            code_lines.append(f"        {result_var} = self.{kernel_name}({inputs_str})")

        # 合并并行结果
        if len(branch_results) > 1:
            code_lines.append(f"        # 合并并行结果")
            merged = branch_results[0]
            for i in range(1, len(branch_results)):
                new_merged = f"merged_parallel_{i}"
                code_lines.append(f"        {new_merged} = self.merge_results({merged}, {branch_results[i]})")
                merged = new_merged
            current_result = merged
        else:
            current_result = branch_results[0]

        # 顺序执行后半部分
        for i, kernel in enumerate(sequential_kernels):
            kernel_name = kernel["name"]
            kernel_inputs = [inp[0] for inp in kernel["inputs"]]

            # 使用前一个结果作为第一个输入
            kernel_inputs[0] = current_result

            result_var = f"sequential_{i}"
            inputs_str = ", ".join(kernel_inputs)
            code_lines.append(f"        {result_var} = self.{kernel_name}({inputs_str})")
            current_result = result_var

        # 返回最终结果
        code_lines.append(f"        return {current_result}")

        return {
            "mode": "mixed",
            "code": "\n".join(code_lines),
            "inputs": input_params,
            "output_shape": output_shape,
            "needs_merge_kernel": len(branch_results) > 1,
        }

    def generate_merge_kernel(self, shape: Tuple[int, int] = (128, 128)) -> str:
        """生成用于合并结果的辅助内核

        Args:
            shape: 张量形状

        Returns:
            合并内核的代码字符串
        """
        rows, cols = shape
        code = f"""    @pl.function(type=pl.FunctionType.InCore)
    def merge_results(self, a: pl.Tensor[[{rows}, {cols}], pl.FP32],
                      b: pl.Tensor[[{rows}, {cols}], pl.FP32],
                      output: pl.Tensor[[{rows}, {cols}], pl.FP32]) -> pl.Tensor[[{rows}, {cols}], pl.FP32]:
        tile_a = pl.op.block.load(a, [0, 0], [{rows}, {cols}])
        tile_b = pl.op.block.load(b, [0, 0], [{rows}, {cols}])
        result_tile = pl.op.block.add(tile_a, tile_b)
        result = pl.op.block.store(result_tile, [0, 0], [{rows}, {cols}], output)
        return result"""
        return code
