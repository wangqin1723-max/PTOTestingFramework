"""
InCore 内核函数生成器

该模块负责生成 @pl.function(type=pl.FunctionType.InCore) 内核函数。
每个内核包含一系列随机生成的算子操作链。
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from .fuzzer import OpFuzzer, is_shape_aligned, generate_aligned_shape


class KernelGenerator:
    """生成 InCore 内核函数的生成器"""

    def __init__(self, seed: Optional[int] = None, enable_advanced_ops: bool = False):
        """初始化内核生成器

        Args:
            seed: 随机种子，用于可重现性
            enable_advanced_ops: 启用高级算子（row_expand, matmul等）
        """
        self.rng = random.Random(seed)
        self.fuzzer = OpFuzzer(seed=seed, enable_advanced_ops=enable_advanced_ops)

    def generate_kernel(
        self,
        kernel_name: str,
        num_inputs: int = 2,
        num_ops: int = 5,
        shape: Tuple[int, int] = (128, 128),
        allow_scalars: bool = True,
        input_shapes: Optional[List[Tuple[int, int]]] = None,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """生成单个 InCore 内核

        Args:
            kernel_name: 内核函数名称
            num_inputs: 输入张量数量（如果未指定 input_shapes）
            num_ops: 操作数量
            shape: 默认张量形状（如果未指定 input_shapes）
            allow_scalars: 是否允许标量操作
            input_shapes: 每个输入的形状列表，如果指定则覆盖 num_inputs 和 shape
            output_shape: 输出形状，如果指定则覆盖默认行为

        Returns:
            包含内核信息的字典:
            - name: 内核名称
            - inputs: 输入参数列表 [(name, shape), ...]
            - output_shape: 输出形状
            - op_chain: 操作链
            - code: 生成的 PyPTO 代码
        """
        # 确定输入形状
        if input_shapes is not None:
            actual_num_inputs = len(input_shapes)
            actual_shapes = input_shapes
        else:
            actual_num_inputs = num_inputs
            actual_shapes = [shape] * num_inputs

        # 验证所有形状是否满足对齐约束
        dtype = "FP32"  # 当前仅支持 FP32
        for i, input_shape in enumerate(actual_shapes):
            if not is_shape_aligned(input_shape, dtype):
                # 如果形状不对齐，使用最接近的对齐形状
                print(f"Warning: Input shape {input_shape} is not 32-byte aligned. Regenerating aligned shape.")
                actual_shapes[i] = generate_aligned_shape(self.rng, dtype)

        # 确定输出形状并验证对齐
        if output_shape is not None:
            actual_output_shape = output_shape
            if not is_shape_aligned(actual_output_shape, dtype):
                print(f"Warning: Output shape {actual_output_shape} is not 32-byte aligned. Regenerating aligned shape.")
                actual_output_shape = generate_aligned_shape(self.rng, dtype)
        else:
            actual_output_shape = actual_shapes[0]

        # 生成操作链
        op_chain = self.fuzzer.generate_op_chain(
            num_ops=num_ops,
            input_count=actual_num_inputs,
            allow_scalars=allow_scalars,
            track_shapes=False,
            default_shape=actual_output_shape,
        )

        # 生成输入参数
        input_names = [chr(97 + i) for i in range(actual_num_inputs)]  # a, b, c, ...
        inputs = [(name, actual_shapes[i]) for i, name in enumerate(input_names)]

        # 生成内核代码
        code = self._generate_kernel_code(
            kernel_name=kernel_name,
            inputs=inputs,
            op_chain=op_chain,
            output_shape=actual_output_shape,
        )

        return {
            "name": kernel_name,
            "inputs": inputs,
            "output_shape": actual_output_shape,
            "op_chain": op_chain,
            "code": code,
        }

    def _generate_kernel_code(
        self,
        kernel_name: str,
        inputs: List[Tuple[str, Tuple[int, int]]],
        op_chain: List[Dict[str, Any]],
        output_shape: Tuple[int, int],
    ) -> str:
        """生成内核函数代码

        Args:
            kernel_name: 内核名称
            inputs: 输入参数列表
            op_chain: 操作链
            output_shape: 输出形状

        Returns:
            生成的 PyPTO 代码字符串
        """
        rows, cols = output_shape

        # 生成函数签名 - 添加 output_tensor 参数
        params = []
        for name, (r, c) in inputs:
            params.append(f"{name}: pl.Tensor[[{r}, {c}], pl.FP32]")
        # 添加 output_tensor 参数
        params.append(f"output: pl.Tensor[[{rows}, {cols}], pl.FP32]")

        code_lines = [
            f"    @pl.function(type=pl.FunctionType.InCore)",
            f"    def {kernel_name}(self, {', '.join(params)}) -> pl.Tensor[[{rows}, {cols}], pl.FP32]:",
        ]

        # 加载输入张量 - 使用输出形状作为加载大小
        for name, (r, c) in inputs:
            code_lines.append(f"        tile_{name} = pl.load({name}, offsets=[0, 0], shapes=[{rows}, {cols}])")

        # 生成操作链
        for op_dict in op_chain:
            op = op_dict["op"]
            inputs_str = ", ".join(op_dict["inputs"])
            output = op_dict["output"]
            params = op_dict.get("params")

            # 去掉 block. 前缀，直接使用 pl.xxx
            op_name = op.name.replace("block.", "")

            if params:
                params_str = ", ".join(f"{k}={v}" for k, v in params.items())
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
            first_input = inputs[0][0]
            code_lines.append(f"        result = pl.store(tile_{first_input}, offsets=[0, 0], shapes=[{rows}, {cols}], output_tensor=output)")
            code_lines.append(f"        return result")

        return "\n".join(code_lines)

    def generate_multiple_kernels(
        self,
        num_kernels: int = 3,
        num_inputs_range: Tuple[int, int] = (2, 3),
        num_ops_range: Tuple[int, int] = (3, 7),
        shape: Tuple[int, int] = (128, 128),
        input_shapes_list: Optional[List[List[Tuple[int, int]]]] = None,
        output_shapes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        """生成多个 InCore 内核

        Args:
            num_kernels: 要生成的内核数量
            num_inputs_range: 输入数量范围 (min, max)
            num_ops_range: 操作数量范围 (min, max)
            shape: 默认张量形状
            input_shapes_list: 每个内核的输入形状列表，如果指定则覆盖其他参数
                              例如: [[(128,128), (64,64)], [(256,256)], ...]
            output_shapes: 每个内核的输出形状列表（可选）

        Returns:
            内核信息字典列表
        """
        kernels = []
        for i in range(num_kernels):
            num_ops = self.rng.randint(*num_ops_range)

            # 确定输入形状
            if input_shapes_list and i < len(input_shapes_list):
                kernel_input_shapes = input_shapes_list[i]
                kernel_output_shape = output_shapes[i] if output_shapes and i < len(output_shapes) else None
                kernel = self.generate_kernel(
                    kernel_name=f"kernel_{i}",
                    num_ops=num_ops,
                    shape=shape,
                    input_shapes=kernel_input_shapes,
                    output_shape=kernel_output_shape,
                )
            else:
                num_inputs = self.rng.randint(*num_inputs_range)
                kernel_output_shape = output_shapes[i] if output_shapes and i < len(output_shapes) else None
                kernel = self.generate_kernel(
                    kernel_name=f"kernel_{i}",
                    num_inputs=num_inputs,
                    num_ops=num_ops,
                    shape=shape,
                    output_shape=kernel_output_shape,
                )
            kernels.append(kernel)

        return kernels
