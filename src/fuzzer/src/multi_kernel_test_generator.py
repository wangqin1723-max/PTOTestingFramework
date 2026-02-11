"""
多内核测试用例生成器

该模块负责生成完整的测试用例，包括：
- 多个 InCore 内核
- Orchestration 组合函数
- NumPy 参考实现
- PTOTestCase 测试类
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .fuzzer import OpFuzzer

from .kernel_generator import KernelGenerator
from .orchestrator_generator import OrchestratorGenerator


class MultiKernelTestGenerator:
    """生成多内核测试用例的生成器"""

    def __init__(self, seed: Optional[int] = None, enable_advanced_ops: bool = False):
        """初始化测试生成器

        Args:
            seed: 随机种子，用于可重现性
            enable_advanced_ops: 启用高级算子（row_expand, matmul等）
        """
        self.seed = seed
        self.enable_advanced_ops = enable_advanced_ops
        self.kernel_gen = KernelGenerator(seed=seed, enable_advanced_ops=enable_advanced_ops)
        self.orch_gen = OrchestratorGenerator(seed=seed)
        self.fuzzer = OpFuzzer(seed=seed, enable_advanced_ops=enable_advanced_ops)

    def _compute_output_shapes_for_sequential(
        self,
        num_kernels: int,
        default_shape: Tuple[int, int],
        input_shapes_list: Optional[List[List[Tuple[int, int]]]],
        mode: str,
    ) -> List[Tuple[int, int]]:
        """计算顺序模式下每个内核的输出形状，确保形状兼容性

        Args:
            num_kernels: 内核数量
            default_shape: 默认形状
            input_shapes_list: 输入形状列表
            mode: 组合模式

        Returns:
            每个内核的输出形状列表
        """
        output_shapes = []

        if mode == "sequential":
            # 顺序模式：kernel_i 的输出必须匹配 kernel_{i+1} 的第一个输入
            for i in range(num_kernels):
                if i == num_kernels - 1:
                    # 最后一个内核：输出形状使用其第一个输入的形状
                    if input_shapes_list and i < len(input_shapes_list):
                        output_shapes.append(input_shapes_list[i][0])
                    else:
                        output_shapes.append(default_shape)
                else:
                    # 非最后一个内核：输出形状必须匹配下一个内核的第一个输入
                    if input_shapes_list and i + 1 < len(input_shapes_list):
                        next_kernel_first_input = input_shapes_list[i + 1][0]
                        output_shapes.append(next_kernel_first_input)
                    else:
                        output_shapes.append(default_shape)

        elif mode == "branching":
            # 分支模式：所有内核必须有相同的输出形状（用于合并）
            # 使用第一个内核的第一个输入形状作为统一输出形状
            if input_shapes_list and len(input_shapes_list) > 0:
                unified_output_shape = input_shapes_list[0][0]
            else:
                unified_output_shape = default_shape

            for i in range(num_kernels):
                output_shapes.append(unified_output_shape)

        elif mode == "mixed":
            # 混合模式：前半部分并行，后半部分顺序
            mid = num_kernels // 2

            # 并行部分：所有内核使用相同的输出形状
            if input_shapes_list and len(input_shapes_list) > 0:
                parallel_output_shape = input_shapes_list[0][0]
            else:
                parallel_output_shape = default_shape

            for i in range(num_kernels):
                if i < mid:
                    # 并行部分：统一输出形状
                    output_shapes.append(parallel_output_shape)
                elif i == mid:
                    # 第一个顺序内核：输出形状匹配下一个内核的第一个输入（如果有）
                    if i == num_kernels - 1:
                        # 如果是最后一个，使用其第一个输入的形状
                        if input_shapes_list and i < len(input_shapes_list):
                            output_shapes.append(input_shapes_list[i][0])
                        else:
                            output_shapes.append(default_shape)
                    else:
                        # 匹配下一个内核的第一个输入
                        if input_shapes_list and i + 1 < len(input_shapes_list):
                            output_shapes.append(input_shapes_list[i + 1][0])
                        else:
                            output_shapes.append(default_shape)
                else:
                    # 后续顺序内核
                    if i == num_kernels - 1:
                        # 最后一个内核
                        if input_shapes_list and i < len(input_shapes_list):
                            output_shapes.append(input_shapes_list[i][0])
                        else:
                            output_shapes.append(default_shape)
                    else:
                        # 匹配下一个内核的第一个输入
                        if input_shapes_list and i + 1 < len(input_shapes_list):
                            output_shapes.append(input_shapes_list[i + 1][0])
                        else:
                            output_shapes.append(default_shape)

        return output_shapes

    def _regenerate_kernel_code_with_unified_shapes(
        self,
        kernel: Dict[str, Any],
        input_shapes_map: Dict[str, Tuple[int, int]],
    ) -> str:
        """使用统一的输入形状重新生成 kernel 代码

        Args:
            kernel: 内核信息字典
            input_shapes_map: 统一的输入形状映射

        Returns:
            重新生成的 kernel 代码
        """
        kernel_name = kernel["name"]
        output_shape = kernel["output_shape"]
        op_chain = kernel["op_chain"]
        rows, cols = output_shape

        # 使用统一的输入形状生成函数签名
        params = []
        for inp_name, _ in kernel["inputs"]:
            unified_shape = input_shapes_map[inp_name]
            params.append(f"{inp_name}: pl.Tensor[[{unified_shape[0]}, {unified_shape[1]}], pl.FP32]")
        # 添加 output_tensor 参数
        params.append(f"output: pl.Tensor[[{rows}, {cols}], pl.FP32]")

        code_lines = [
            f"    @pl.function(type=pl.FunctionType.InCore)",
            f"    def {kernel_name}(self, {', '.join(params)}) -> pl.Tensor[[{rows}, {cols}], pl.FP32]:",
        ]

        # 加载输入张量 - 使用每个输入的实际定义形状
        for inp_name, _ in kernel["inputs"]:
            inp_shape = input_shapes_map[inp_name]
            code_lines.append(f"        tile_{inp_name} = pl.load({inp_name}, offsets=[0, 0], shapes=[{inp_shape[0]}, {inp_shape[1]}])")

        # 生成操作链
        for op_dict in op_chain:
            op = op_dict["op"]
            inputs_str = ", ".join(op_dict["inputs"])
            output = op_dict["output"]
            params_dict = op_dict.get("params")

            # 去掉 block. 前缀，直接使用 pl.xxx
            op_name = op.name.replace("block.", "")

            if params_dict:
                params_str = ", ".join(f"{k}={v}" for k, v in params_dict.items())
                code_lines.append(f"        {output} = pl.{op_name}({inputs_str}, {params_str})")
            else:
                code_lines.append(f"        {output} = pl.{op_name}({inputs_str})")

        # Store 结果并返回
        if op_chain:
            last_output = op_chain[-1]["output"]
            code_lines.append(f"        result = pl.store({last_output}, offsets=[0, 0], shapes=[{rows}, {cols}], output_tensor=output)")
            code_lines.append(f"        return result")
        else:
            # 如果没有操作，直接 store 第一个输入
            first_input = kernel["inputs"][0][0]
            code_lines.append(f"        result = pl.store(tile_{first_input}, offsets=[0, 0], shapes=[{rows}, {cols}], output_tensor=output)")
            code_lines.append(f"        return result")

        return "\n".join(code_lines)

    def generate_test_case(
        self,
        test_name: str,
        num_kernels: int = 3,
        orchestration_mode: str = "sequential",
        shape: Tuple[int, int] = (128, 128),
        num_ops_range: Tuple[int, int] = (3, 7),
        input_shapes_list: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> str:
        """生成完整的测试用例代码

        Args:
            test_name: 测试用例名称
            num_kernels: 内核数量
            orchestration_mode: 组合模式 ("sequential", "branching", "mixed")
            shape: 张量形状
            num_ops_range: 每个内核的操作数量范围
            input_shapes_list: 每个内核的输入形状列表（可选）

        Returns:
            完整的测试用例代码字符串
        """
        # 对于 sequential、branching 和 mixed 模式，计算输出形状以确保兼容性
        if orchestration_mode in ["sequential", "branching", "mixed"]:
            output_shapes = self._compute_output_shapes_for_sequential(
                num_kernels, shape, input_shapes_list, orchestration_mode
            )
        else:
            output_shapes = None

        # 生成多个内核
        kernels = self.kernel_gen.generate_multiple_kernels(
            num_kernels=num_kernels,
            num_inputs_range=(2, 3),
            num_ops_range=num_ops_range,
            shape=shape,
            input_shapes_list=input_shapes_list,
            output_shapes=output_shapes,
        )

        # 生成 Orchestration 函数
        if orchestration_mode == "sequential":
            orch_info = self.orch_gen.generate_sequential(kernels, shape)
        elif orchestration_mode == "branching":
            orch_info = self.orch_gen.generate_branching(kernels, shape)
        elif orchestration_mode == "mixed":
            orch_info = self.orch_gen.generate_mixed(kernels, shape)
        else:
            raise ValueError(f"未知的组合模式: {orchestration_mode}")

        # 生成 NumPy 参考实现
        numpy_code = self._generate_numpy_reference(kernels, orch_info)

        # 生成完整的测试类
        test_code = self._generate_test_class(
            test_name=test_name,
            kernels=kernels,
            orch_info=orch_info,
            numpy_code=numpy_code,
            shape=shape,
        )

        return test_code

    def _generate_numpy_reference(
        self,
        kernels: List[Dict[str, Any]],
        orch_info: Dict[str, Any],
    ) -> str:
        """生成 NumPy 参考实现代码

        Args:
            kernels: 内核信息列表
            orch_info: Orchestration 信息

        Returns:
            NumPy 参考实现代码字符串
        """
        code_lines = []

        # 为每个内核生成 NumPy 函数
        for kernel in kernels:
            kernel_name = kernel["name"]
            input_names = [inp[0] for inp in kernel["inputs"]]
            op_chain = kernel["op_chain"]

            # 嵌套函数不需要 self 参数
            code_lines.append(f"    def _numpy_{kernel_name}({', '.join(input_names)}):")
            code_lines.append(f"        \"\"\"NumPy 实现: {kernel_name}\"\"\"")

            # 生成 NumPy 操作
            code_lines.append(f"        # 创建变量环境")
            code_lines.append(f"        env = {{}}")
            for name in input_names:
                code_lines.append(f"        env['tile_{name}'] = {name}.copy()")

            code_lines.append(f"")
            code_lines.append(f"        # 执行操作链")
            for op_dict in op_chain:
                op = op_dict["op"]
                inputs = op_dict["inputs"]
                output = op_dict["output"]

                # 获取输入值
                input_vals = []
                for inp in inputs:
                    if inp.startswith("tile_") or inp.startswith("tmp_"):
                        input_vals.append(f"env['{inp}']")
                    else:
                        input_vals.append(inp)

                # 应用约束
                if "avoid_zero" in op.constraints and op.constraints["avoid_zero"]:
                    for i, inp in enumerate(inputs):
                        if inp.startswith("tile_") or inp.startswith("tmp_"):
                            code_lines.append(f"        env['{inp}'] = np.where(np.abs(env['{inp}']) < 0.01, 1.0, env['{inp}'])")

                if "positive_only" in op.constraints and op.constraints["positive_only"]:
                    for i, inp in enumerate(inputs):
                        if inp.startswith("tile_") or inp.startswith("tmp_"):
                            code_lines.append(f"        env['{inp}'] = np.abs(env['{inp}']) + 1e-6")

                # 生成操作
                if op.np_equivalent:
                    np_expr = self._get_numpy_operation(op.name, input_vals)
                    code_lines.append(f"        env['{output}'] = {np_expr}")

            code_lines.append(f"        return env['{op_chain[-1]['output']}']")
            code_lines.append(f"")

        return "\n".join(code_lines)

    def _get_numpy_operation(self, op_name: str, input_vals: List[str]) -> str:
        """将 PyPTO 操作名转换为 NumPy 操作表达式

        Args:
            op_name: PyPTO 操作名 (如 "block.add")
            input_vals: 输入值列表

        Returns:
            NumPy 操作表达式字符串
        """
        # 根据操作类型生成表达式
        # 二元操作
        if op_name == "block.add":
            return f"{input_vals[0]} + {input_vals[1]}"
        elif op_name == "block.sub":
            return f"{input_vals[0]} - {input_vals[1]}"
        elif op_name == "block.mul":
            return f"{input_vals[0]} * {input_vals[1]}"
        elif op_name == "block.div":
            return f"{input_vals[0]} / {input_vals[1]}"
        elif op_name == "block.maximum":
            return f"np.maximum({input_vals[0]}, {input_vals[1]})"
        elif op_name == "block.minimum":
            return f"np.minimum({input_vals[0]}, {input_vals[1]})"
        # 标量操作
        elif op_name == "block.adds":
            return f"{input_vals[0]} + {input_vals[1]}"
        elif op_name == "block.subs":
            return f"{input_vals[0]} - {input_vals[1]}"
        elif op_name == "block.muls":
            return f"{input_vals[0]} * {input_vals[1]}"
        elif op_name == "block.divs":
            return f"{input_vals[0]} / {input_vals[1]}"
        # 一元操作
        elif op_name == "block.sqrt":
            return f"np.sqrt({input_vals[0]})"
        elif op_name == "block.rsqrt":
            return f"1.0 / np.sqrt({input_vals[0]})"
        elif op_name == "block.exp":
            return f"np.exp(np.clip({input_vals[0]}, -10, 10))"
        elif op_name == "block.neg":
            return f"-{input_vals[0]}"
        elif op_name == "block.recip":
            return f"1.0 / {input_vals[0]}"
        elif op_name == "block.log":
            return f"np.log({input_vals[0]})"
        elif op_name == "block.abs":
            return f"np.abs({input_vals[0]})"
        elif op_name == "block.relu":
            return f"np.maximum(0, {input_vals[0]})"
        # Row expand 操作
        elif op_name == "block.row_expand_add":
            return f"{input_vals[0]} + {input_vals[1]}"  # Broadcasting
        elif op_name == "block.row_expand_sub":
            return f"{input_vals[0]} - {input_vals[1]}"
        elif op_name == "block.row_expand_mul":
            return f"{input_vals[0]} * {input_vals[1]}"
        elif op_name == "block.row_expand_div":
            return f"{input_vals[0]} / {input_vals[1]}"
        # 矩阵操作
        elif op_name == "block.matmul":
            return f"{input_vals[0]} @ {input_vals[1]}"
        else:
            return f"# 未知操作: {op_name}"

    def _generate_test_class(
        self,
        test_name: str,
        kernels: List[Dict[str, Any]],
        orch_info: Dict[str, Any],
        numpy_code: str,
        shape: Tuple[int, int],
    ) -> str:
        """生成完整的测试类代码

        Args:
            test_name: 测试名称
            kernels: 内核信息列表
            orch_info: Orchestration 信息
            numpy_code: NumPy 参考实现代码
            shape: 张量形状

        Returns:
            完整的测试类代码
        """
        rows, cols = shape
        class_name = f"Test{test_name.replace('_', ' ').title().replace(' ', '')}"

        # 收集所有输入及其实际形状
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

        input_list = sorted(input_shapes_map.keys())

        # 输出形状使用最后一个内核的输出形状
        output_shape = kernels[-1]["output_shape"] if kernels else shape

        # 生成头部
        code_lines = [
            f"class {class_name}(PTOTestCase):",
            f"    \"\"\"",
            f"    测试用例: {test_name}",
            f"    组合模式: {orch_info['mode']}",
            f"    内核数量: {len(kernels)}",
            f"    \"\"\"",
            f"",
            f"    def __init__(self, **kwargs):",
            f"        super().__init__(**kwargs)",
            f"        self.rows = {rows}",
            f"        self.cols = {cols}",
            f"",
            f"    def get_name(self) -> str:",
            f"        return '{test_name}'",
            f"",
            f"    def define_tensors(self) -> List[TensorSpec]:",
            f"        return [",
        ]

        # 定义输入张量 - 使用实际形状
        for inp_name in input_list:
            init_val = 2.0 + input_list.index(inp_name) * 0.5
            inp_shape = input_shapes_map[inp_name]
            code_lines.append(f"            TensorSpec('{inp_name}', [{inp_shape[0]}, {inp_shape[1]}], DataType.FP32, init_value={init_val}),")

        # 定义输出张量 - 使用实际输出形状
        code_lines.append(f"            TensorSpec('output', [{output_shape[0]}, {output_shape[1]}], DataType.FP32, is_output=True),")
        code_lines.append(f"        ]")
        code_lines.append(f"")

        # 生成 PyPTO 程序
        code_lines.append(f"    def get_program(self) -> Any:")
        code_lines.append(f"        import pypto.language as pl")
        code_lines.append(f"")
        code_lines.append(f"        @pl.program")
        code_lines.append(f"        class {test_name.replace('_', ' ').title().replace(' ', '')}Program:")

        # 添加所有内核（需要额外缩进）
        for kernel in kernels:
            # 使用统一的输入形状重新生成 kernel 代码
            regenerated_code = self._regenerate_kernel_code_with_unified_shapes(kernel, input_shapes_map)
            # 为内核代码添加额外的8个空格缩进（4个用于get_program方法，4个用于@pl.program类）
            kernel_lines = regenerated_code.split("\n")
            for line in kernel_lines:
                code_lines.append(f"        {line}")
            code_lines.append(f"")

        # 添加合并内核（如果需要）
        if orch_info.get("needs_merge_kernel", False):
            merge_code = self.orch_gen.generate_merge_kernel(shape)
            merge_lines = merge_code.split("\n")
            for line in merge_lines:
                code_lines.append(f"        {line}")
            code_lines.append(f"")

        # 添加 Orchestration 函数
        orch_lines = orch_info["code"].split("\n")
        for line in orch_lines:
            code_lines.append(f"        {line}")
        code_lines.append(f"")

        code_lines.append(f"        return {test_name.replace('_', ' ').title().replace(' ', '')}Program")
        code_lines.append(f"")

        # 添加 NumPy 参考实现
        code_lines.append(f"    def compute_expected(self, tensors, params=None):")
        code_lines.append(f"        \"\"\"使用 NumPy 计算期望输出\"\"\"")
        # numpy_code 包含嵌套函数定义，需要添加到 compute_expected 内部，所以需要额外缩进
        numpy_lines = numpy_code.split('\n')
        for line in numpy_lines:
            if line.strip():  # 跳过空行
                code_lines.append(f"    {line}")  # 添加额外的4个空格缩进
            else:
                code_lines.append(line)
        code_lines.append(f"")

        # 根据组合模式生成计算逻辑
        if orch_info["mode"] == "sequential":
            code_lines.append(f"        # 顺序执行模式")
            result_var = None
            for i, kernel in enumerate(kernels):
                kernel_name = kernel["name"]
                kernel_inputs = [inp[0] for inp in kernel["inputs"]]

                if i > 0 and result_var:
                    # 第一个输入使用前一个结果（变量名）
                    kernel_inputs[0] = result_var
                    # 构建参数列表：第一个是变量，其他从 tensors 获取
                    inputs_parts = [kernel_inputs[0]]
                    for inp in kernel_inputs[1:]:
                        inputs_parts.append(f"tensors['{inp}']")
                    inputs_str = ", ".join(inputs_parts)
                else:
                    # 第一个内核，所有输入都从 tensors 获取
                    inputs_str = ", ".join([f"tensors['{inp}']" for inp in kernel_inputs])

                result_var = f"result_{i}"
                # 调用嵌套函数不需要 self
                code_lines.append(f"        {result_var} = _numpy_{kernel_name}({inputs_str})")

            code_lines.append(f"        tensors['output'][:] = {result_var}")

        elif orch_info["mode"] == "branching":
            code_lines.append(f"        # 分支执行模式")
            branch_results = []
            for i, kernel in enumerate(kernels):
                kernel_name = kernel["name"]
                kernel_inputs = [inp[0] for inp in kernel["inputs"]]
                result_var = f"branch_{i}"
                branch_results.append(result_var)

                inputs_str = ", ".join([f"tensors['{inp}']" for inp in kernel_inputs])
                # 调用嵌套函数不需要 self
                code_lines.append(f"        {result_var} = _numpy_{kernel_name}({inputs_str})")

            # 合并结果
            if len(branch_results) == 1:
                code_lines.append(f"        tensors['output'][:] = {branch_results[0]}")
            else:
                merged = branch_results[0]
                for i in range(1, len(branch_results)):
                    new_merged = f"merged_{i}"
                    code_lines.append(f"        {new_merged} = {merged} + {branch_results[i]}")
                    merged = new_merged
                code_lines.append(f"        tensors['output'][:] = {merged}")

        elif orch_info["mode"] == "mixed":
            code_lines.append(f"        # 混合执行模式")
            mid = len(kernels) // 2
            parallel_kernels = kernels[:mid]
            sequential_kernels = kernels[mid:]

            # 并行部分
            branch_results = []
            for i, kernel in enumerate(parallel_kernels):
                kernel_name = kernel["name"]
                kernel_inputs = [inp[0] for inp in kernel["inputs"]]
                result_var = f"parallel_{i}"
                branch_results.append(result_var)

                inputs_str = ", ".join([f"tensors['{inp}']" for inp in kernel_inputs])
                # 调用嵌套函数不需要 self
                code_lines.append(f"        {result_var} = _numpy_{kernel_name}({inputs_str})")

            # 合并并行结果
            if len(branch_results) > 1:
                merged = branch_results[0]
                for i in range(1, len(branch_results)):
                    new_merged = f"merged_parallel_{i}"
                    code_lines.append(f"        {new_merged} = {merged} + {branch_results[i]}")
                    merged = new_merged
                current_result = merged
            else:
                current_result = branch_results[0]

            # 顺序部分
            for i, kernel in enumerate(sequential_kernels):
                kernel_name = kernel["name"]
                kernel_inputs = [inp[0] for inp in kernel["inputs"]]
                kernel_inputs[0] = current_result

                result_var = f"sequential_{i}"
                # 第一个输入是变量，其他是张量
                inputs_parts = [kernel_inputs[0]]
                for inp in kernel_inputs[1:]:
                    inputs_parts.append(f"tensors['{inp}']")
                inputs_str = ", ".join(inputs_parts)
                # 调用嵌套函数不需要 self
                code_lines.append(f"        {result_var} = _numpy_{kernel_name}({inputs_str})")
                current_result = result_var

            code_lines.append(f"        tensors['output'][:] = {current_result}")

        code_lines.append(f"")

        return "\n".join(code_lines)

    def generate_test_file(
        self,
        output_path: str,
        test_configs: List[Dict[str, Any]],
    ) -> None:
        """生成完整的测试文件

        Args:
            output_path: 输出文件路径
            test_configs: 测试配置列表，每个配置包含:
                - name: 测试名称
                - num_kernels: 内核数量
                - mode: 组合模式
                - shape: 张量形状
                - num_ops_range: 操作数量范围
        """
        # 生成文件头
        header = '''"""
自动生成的多内核模糊测试用例

该文件由 MultiKernelTestGenerator 自动生成。
包含多个测试用例，每个测试用例包含多个 InCore 内核和一个 Orchestration 函数。
"""

import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from pto_test.core.test_case import DataType, PTOTestCase, TensorSpec

# 添加 pypto 到路径
_FRAMEWORK_ROOT = Path(__file__).parent.parent.parent.parent
_PYPTO_ROOT = _FRAMEWORK_ROOT / "3rdparty" / "pypto" / "python"
if _PYPTO_ROOT.exists() and str(_PYPTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYPTO_ROOT))


'''

        # 生成所有测试用例
        test_cases = []
        for config in test_configs:
            test_code = self.generate_test_case(
                test_name=config["name"],
                num_kernels=config.get("num_kernels", 3),
                orchestration_mode=config.get("mode", "sequential"),
                shape=config.get("shape", (128, 128)),
                num_ops_range=config.get("num_ops_range", (3, 7)),
                input_shapes_list=config.get("input_shapes_list"),
            )
            test_cases.append(test_code)

        # 生成测试套件
        test_suite = '''

class TestMultiKernelFuzzing:
    """多内核模糊测试套件"""

'''

        for config in test_configs:
            test_name = config["name"]
            class_name = f"Test{test_name.replace('_', ' ').title().replace(' ', '')}"
            test_suite += f'''    def test_{test_name}(self, test_runner):
        """测试 {test_name}"""
        test_case = {class_name}()
        result = test_runner.run(test_case)
        assert result.passed, f"测试失败: {{result.error}}"

'''

        # 组合完整文件
        full_content = header + "\n\n".join(test_cases) + test_suite

        # 写入文件
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(full_content, encoding="utf-8")

        print(f"测试文件已生成: {output_path}")
