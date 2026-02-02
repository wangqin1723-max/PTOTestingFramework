"""
Test runner for executing PTO test cases.

Orchestrates the full test execution pipeline:
1. Get program from test case (@pl.program or IRBuilder)
2. Generate kernel code via PyPTO (PassManager -> CceCodegen)
3. Generate orchestration code (auto or custom)
4. Generate kernel_config.py and golden.py
5. Execute via simpler's CodeRunner
6. Validate results
"""

import json
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from pto_test.core.test_case import PTOTestCase, TestConfig, TestResult

# Add pypto and simpler to path
_FRAMEWORK_ROOT = Path(__file__).parent.parent.parent.parent
_PYPTO_ROOT = _FRAMEWORK_ROOT / "3rdparty" / "pypto" / "python"
_SIMPLER_ROOT = _FRAMEWORK_ROOT / "3rdparty" / "simpler"
_SIMPLER_PYTHON = _SIMPLER_ROOT / "python"
_SIMPLER_SCRIPTS = _SIMPLER_ROOT / "examples" / "scripts"

for path in [_PYPTO_ROOT, _SIMPLER_PYTHON, _SIMPLER_SCRIPTS]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Session-level output directory (shared across all tests in a pytest session)
_SESSION_OUTPUT_DIR = None


def _get_session_output_dir() -> Path:
    """Get or create session-level output directory with timestamp.

    Returns:
        Path to the session output directory (build/outputs/output_{timestamp}/).
    """
    global _SESSION_OUTPUT_DIR
    if _SESSION_OUTPUT_DIR is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = _FRAMEWORK_ROOT / "build" / "outputs"
        _SESSION_OUTPUT_DIR = output_base / f"output_{timestamp}"
        _SESSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSION_OUTPUT_DIR


class TestRunner:
    """Executes PTO test cases via simpler's CodeRunner.

    This runner integrates with simpler's CodeRunner to execute tests:
    1. Generate kernel C++ from PyPTO program via codegen module
    2. Generate orchestration C++ (auto or custom)
    3. Generate kernel_config.py and golden.py
    4. Use CodeRunner to compile, execute, and validate

    Example:
        runner = TestRunner(TestConfig(platform="a2a3sim"))
        result = runner.run(my_test_case)
        assert result.passed
    """

    def __init__(self, config: Optional[TestConfig] = None):
        """Initialize test runner.

        Args:
            config: Test configuration. If None, uses default config.
        """
        self.config = config or TestConfig()
        self._initialized = False

    def run(self, test_case: PTOTestCase) -> TestResult:
        """Run a test case and return results.

        Args:
            test_case: The test case to run.

        Returns:
            TestResult with pass/fail status and details.
        """
        start_time = time.time()
        test_name = test_case.get_name()

        # Determine work directory based on save_kernels configuration
        if self.config.save_kernels:
            # Always save mode: use persistent directory directly
            if self.config.save_kernels_dir:
                work_dir = Path(self.config.save_kernels_dir) / test_name
            else:
                session_dir = _get_session_output_dir()
                work_dir = session_dir / test_name
            work_dir.mkdir(parents=True, exist_ok=True)
            use_temp = False
        else:
            # Temporary mode: use temp directory for execution
            work_dir = Path(tempfile.mkdtemp(prefix=f"pto_test_{test_name}_"))
            use_temp = True

        try:
            # Import codegen modules
            from pto_test.codegen.kernel_generator import KernelGenerator
            from pto_test.codegen.orch_generator import OrchGenerator
            from pto_test.codegen.config_generator import ConfigGenerator
            from pto_test.codegen.golden_generator import GoldenGenerator

            kernels_dir = work_dir / "kernels"
            kernels_dir.mkdir(parents=True, exist_ok=True)

            # 1. Generate kernel C++ files
            program = test_case.get_program()
            if program is None:
                raise ValueError(
                    f"Test case {test_name} must implement get_program() "
                    "to return a @pl.program class or ir.Program"
                )

            strategy = test_case.get_strategy()
            kernel_gen = KernelGenerator(strategy=strategy)
            kernel_configs = kernel_gen.generate(
                program,
                kernels_dir,
                dump_passes=self.config.dump_passes,
            )

            if not kernel_configs:
                raise ValueError(f"No kernels generated for {test_name}")

            # Move pass_dump to same level as metadata.json if it exists
            pass_dump_in_kernels = kernels_dir / "pass_dump"
            if pass_dump_in_kernels.exists():
                pass_dump_target = work_dir / "pass_dump"
                # Remove target if it exists to avoid nested directories
                if pass_dump_target.exists():
                    shutil.rmtree(pass_dump_target)
                shutil.move(str(pass_dump_in_kernels), str(pass_dump_target))

            # 2. Generate orchestration C++ file
            orch_dir = kernels_dir / "orchestration"
            orch_dir.mkdir()

            orch_code = test_case.get_orchestration()
            if orch_code is None:
                # Auto-generate orchestration
                orch_gen = OrchGenerator()
                orch_code = orch_gen.generate(
                    test_case.tensor_specs, kernel_configs
                )

            orch_path = orch_dir / "orch.cpp"
            orch_path.write_text(orch_code)

            # 3. Generate kernel_config.py
            config_gen = ConfigGenerator()
            config_gen.write(
                kernels_dir,
                kernel_configs,
                str(orch_path),
                "build_test_graph",
            )

            # 4. Generate golden.py with inline compute logic
            golden_path = kernels_dir / "golden.py"
            golden_gen = GoldenGenerator()
            golden_gen.write(test_case, golden_path)

            # 5. Write metadata.json if saving
            if self.config.save_kernels:
                metadata = {
                    "test_name": test_name,
                    "strategy": str(strategy),
                    "dump_passes": self.config.dump_passes,
                    "timestamp": datetime.now().isoformat(),
                    "kernels": [
                        {
                            "func_id": k["func_id"],
                            "function_name": k.get("function_name", "unknown"),
                            "core_type": k["core_type"],
                            "source": str(Path("kernels") / Path(k["source"]).relative_to(kernels_dir)),
                        }
                        for k in kernel_configs
                    ],
                }
                (work_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2, default=str)
                )

            # 6. Execute via CodeRunner (skip if codegen_only)
            if self.config.codegen_only:
                # Codegen-only mode: skip runtime execution
                return TestResult(
                    passed=True,
                    test_name=test_name,
                    execution_time=time.time() - start_time,
                )

            self._execute_with_code_runner(
                kernels_dir, golden_path, test_name
            )

            return TestResult(
                passed=True,
                test_name=test_name,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            import traceback
            return TestResult(
                passed=False,
                test_name=test_name,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                execution_time=time.time() - start_time,
            )
        finally:
            # Clean up temporary directory if used
            if use_temp and work_dir.exists():
                shutil.rmtree(work_dir)

    def _execute_with_code_runner(
        self,
        kernels_dir: Path,
        golden_path: Path,
        test_name: str,
    ) -> None:
        """Execute test using simpler's CodeRunner.

        Args:
            kernels_dir: Path to directory with kernel_config.py
            golden_path: Path to golden.py
            test_name: Name of the test (for logging)

        Raises:
            Exception: If test execution fails
        """
        from code_runner import CodeRunner

        runner = CodeRunner(
            kernels_dir=str(kernels_dir),
            golden_path=str(golden_path),
            platform=self.config.platform,
            device_id=self.config.device_id,
        )

        # Run the test
        runner.run()


class TestSuite:
    """Collection of test cases that can be run together."""

    def __init__(self, name: str, config: Optional[TestConfig] = None):
        """Initialize test suite.

        Args:
            name: Suite name.
            config: Configuration for all tests in suite.
        """
        self.name = name
        self.config = config or TestConfig()
        self._test_cases: list = []

    def add_test(self, test_case: PTOTestCase) -> "TestSuite":
        """Add a test case to the suite."""
        self._test_cases.append(test_case)
        return self

    def run_all(self, runner: Optional[TestRunner] = None) -> Dict[str, TestResult]:
        """Run all test cases in the suite."""
        if runner is None:
            runner = TestRunner(self.config)

        results = {}
        for test_case in self._test_cases:
            result = runner.run(test_case)
            results[test_case.get_name()] = result
            print(result)

        return results

    def summary(self, results: Dict[str, TestResult]) -> str:
        """Generate summary of test results."""
        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        failed = total - passed

        lines = [
            f"\n{'=' * 50}",
            f"Test Suite: {self.name}",
            f"{'=' * 50}",
            f"Passed: {passed}/{total}",
            f"Failed: {failed}/{total}",
        ]

        if failed > 0:
            lines.append("\nFailed tests:")
            for name, result in results.items():
                if not result.passed:
                    lines.append(f"  - {name}: {result.error}")

        return "\n".join(lines)
