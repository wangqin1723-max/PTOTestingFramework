"""
Code generation module for PTO testing framework.

This module provides generators that convert PyPTO programs and test specifications
into the files required by simpler's CodeRunner:
- kernel_generator: PyPTO Program -> CCE C++ kernel files
- orch_generator: TensorSpec[] -> orchestration C++ code
- config_generator: kernel configs -> kernel_config.py
- golden_generator: PTOTestCase -> golden.py
"""

from pto_test.codegen.kernel_generator import KernelGenerator
from pto_test.codegen.orch_generator import OrchGenerator
from pto_test.codegen.config_generator import ConfigGenerator
from pto_test.codegen.golden_generator import GoldenGenerator

__all__ = [
    "KernelGenerator",
    "OrchGenerator",
    "ConfigGenerator",
    "GoldenGenerator",
]
