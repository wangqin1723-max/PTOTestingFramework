"""
pytest configuration for generated multi-kernel fuzz tests.

This conftest imports all fixtures from the main tests/conftest.py
to ensure generated tests have access to the same CLI options and fixtures.
"""

import sys
from pathlib import Path

# Add framework root to path
_FRAMEWORK_ROOT = Path(__file__).parent.parent.parent.parent
_TESTS_DIR = _FRAMEWORK_ROOT / "tests"

if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Import all fixtures and configuration from main conftest
from tests.conftest import (
    pytest_addoption,
    pytest_configure,
    pytest_collection_modifyitems,
    test_config,
    test_runner,
    optimization_strategy,
    fuzz_count,
    fuzz_seed,
    tensor_shape,
    STANDARD_SHAPES,
)

__all__ = [
    'pytest_addoption',
    'pytest_configure',
    'pytest_collection_modifyitems',
    'test_config',
    'test_runner',
    'optimization_strategy',
    'fuzz_count',
    'fuzz_seed',
    'tensor_shape',
    'STANDARD_SHAPES',
]
