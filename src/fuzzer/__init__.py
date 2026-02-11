"""
Multi-kernel fuzzing framework for PyPTO programs.

This is the main entry point for the fuzzer framework.
External users should import from this module.

Example:
    from fuzzer import OpFuzzer, MultiKernelTestGenerator

    # Create a test generator
    generator = MultiKernelTestGenerator(seed=42)

    # Generate a test case
    test_code = generator.generate_test_case(
        class_name="TestMyFuzz",
        num_kernels=3,
        ops_per_kernel=(2, 5),
        composition_style="sequential"
    )
"""

# Import from internal src module
from .src import (
    OpFuzzer,
    OpSpec,
    KernelGenerator,
    OrchestratorGenerator,
    MultiKernelTestGenerator,
)

__all__ = [
    "OpFuzzer",
    "OpSpec",
    "KernelGenerator",
    "OrchestratorGenerator",
    "MultiKernelTestGenerator",
]

__version__ = "1.0.0"
