"""
多内核模糊测试框架使用示例

该脚本演示如何使用多内核测试生成器创建测试用例。
支持通过命令行参数控制生成的测试用例数量和配置。

使用方法:
    python example_multi_kernel.py --num-cases 5
"""

import argparse
import sys
from pathlib import Path

# 添加当前目录到路径
_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from src.multi_kernel_test_generator import MultiKernelTestGenerator


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="生成多内核模糊测试用例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成默认的5个测试用例
  python example_multi_kernel.py

  # 生成3个测试用例
  python example_multi_kernel.py --num-cases 3

  # 指定输出文件
  python example_multi_kernel.py --output custom_test.py
        """
    )

    parser.add_argument(
        "--num-cases",
        type=int,
        default=1,
        choices=range(1, 6),
        metavar="N",
        help="生成的测试用例数量 (1-5，默认: 5)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径 (默认: src/fuzzer/generated_tests/test_fuzz_multi_kernel.py)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=4,
        help="随机种子，用于可重现性 (默认: 42)"
    )

    parser.add_argument(
        "--enable-advanced-ops",
        action="store_true",
        help="启用高级算子 (row_expand, matmul等)"
    )

    args = parser.parse_args()

    # 设置输出路径
    if args.output:
        output_path = args.output
    else:
        output_path = str(_SCRIPT_DIR / "generated_tests" / "test_fuzz_multi_kernel.py")

    print(f"多内核模糊测试生成器")
    print(f"=" * 60)
    print(f"测试用例数量: {args.num_cases}")
    print(f"随机种子: {args.seed}")
    print(f"启用高级算子: {'是' if args.enable_advanced_ops else '否'}")
    print(f"输出文件: {output_path}")
    print(f"=" * 60)
    print()

    # 定义5种不同配置的测试用例
    all_configs = [
        {
            "name": "fuzz_sequential_simple",
            "num_kernels": 2,
            "mode": "sequential",
            "shape": (128, 128),
            "num_ops_range": (3, 5),
            "input_shapes_list": [
                [(128, 128), (128, 128)],  # kernel_0: 2个相同维度的输入
                [(128, 128), (128, 128), (128, 128)],  # kernel_1: 3个相同维度的输入
            ],
            "description": "简单顺序执行：2个内核，相同维度输入"
        },
        {
            "name": "fuzz_branching_parallel",
            "num_kernels": 3,
            "mode": "branching",
            "shape": (128, 128),
            "num_ops_range": (4, 6),
            "input_shapes_list": [
                [(128, 128), (128, 128)],  # kernel_0: 2个相同维度
                [(128, 128), (128, 128)],  # kernel_1: 2个相同维度
                [(128, 128)],              # kernel_2: 1个输入
            ],
            "description": "分支并行执行：3个内核，相同维度输入"
        },
        {
            "name": "fuzz_mixed_complex",
            "num_kernels": 4,
            "mode": "mixed",
            "shape": (128, 128),
            "num_ops_range": (5, 8),
            "input_shapes_list": None,  # 使用随机生成
            "description": "混合模式：前2个并行，后2个顺序，随机输入"
        },
        {
            "name": "fuzz_sequential_deep",
            "num_kernels": 5,
            "mode": "sequential",
            "shape": (128, 128),
            "num_ops_range": (6, 10),
            "input_shapes_list": None,  # 使用随机生成
            "description": "深度顺序执行：5个内核链式调用，随机输入"
        },
        {
            "name": "fuzz_branching_wide",
            "num_kernels": 4,
            "mode": "branching",
            "shape": (128, 128),
            "num_ops_range": (4, 7),
            "input_shapes_list": [
                [(128, 128), (128, 128), (128, 128)],  # kernel_0: 3个相同维度
                [(128, 128)],                          # kernel_1: 1个输入
                [(128, 128), (128, 128)],              # kernel_2: 2个相同维度
                [(128, 128), (128, 128)],              # kernel_3: 2个相同维度
            ],
            "description": "宽分支执行：4个内核，统一维度输入"
        },
    ]

    # 根据 num_cases 选择配置
    selected_configs = all_configs[:args.num_cases]

    print("将生成以下测试用例:")
    print()
    for i, config in enumerate(selected_configs, 1):
        print(f"{i}. {config['name']}")
        print(f"   {config['description']}")
        print()

    # 创建生成器
    generator = MultiKernelTestGenerator(seed=args.seed, enable_advanced_ops=args.enable_advanced_ops)

    # 生成测试文件
    print("正在生成测试文件...")
    generator.generate_test_file(
        output_path=output_path,
        test_configs=selected_configs,
    )

    print()
    print(f"✓ 成功生成 {args.num_cases} 个测试用例")
    print(f"✓ 输出文件: {output_path}")
    print()
    print("运行测试:")
    print(f"  pytest {output_path}")
    print()


if __name__ == "__main__":
    main()
