"""
PTO Testing Framework

End-to-end testing framework for PyPTO frontend and Simpler runtime.
"""

from pto_test.core.test_case import (
    PTOTestCase,
    TensorSpec,
    TestConfig,
    TestResult,
    DataType,
)
from pto_test.core.test_runner import TestRunner, TestSuite
from pto_test.core.validators import ResultValidator

# Codegen module exports
from pto_test.codegen import (
    KernelGenerator,
    OrchGenerator,
    ConfigGenerator,
    GoldenGenerator,
)

__version__ = "0.1.0"
__all__ = [
    # Core
    "PTOTestCase",
    "TensorSpec",
    "TestConfig",
    "TestResult",
    "DataType",
    "TestRunner",
    "TestSuite",
    "ResultValidator",
    # Codegen
    "KernelGenerator",
    "OrchGenerator",
    "ConfigGenerator",
    "GoldenGenerator",
]
