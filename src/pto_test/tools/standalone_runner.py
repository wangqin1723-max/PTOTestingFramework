#!/usr/bin/env python3
"""
Standalone runner for testing manually completed orchestration code.

This tool allows you to:
1. Generate skeleton orchestration code using codegen_only mode
2. Manually complete the task creation logic in the orchestration file
3. Run the completed code on the device for testing

Usage:
    # Step 1: Generate skeleton code (codegen only)
    python -m pto_test.tools.standalone_runner --codegen-only --test-dir ./my_test

    # Step 2: Edit the generated orchestration file at:
    #         ./my_test/kernels/orchestration/orch.cpp

    # Step 3: Run the completed code on device
    python -m pto_test.tools.standalone_runner --run --test-dir ./my_test --platform a2a3sim

You can also use it programmatically:
    from pto_test.tools.standalone_runner import StandaloneRunner

    runner = StandaloneRunner()
    runner.run_completed_test('./my_test', platform='a2a3sim')
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add required paths
_SCRIPT_DIR = Path(__file__).parent
_FRAMEWORK_ROOT = _SCRIPT_DIR.parent.parent.parent
_PYPTO_ROOT = _FRAMEWORK_ROOT / "3rdparty" / "pypto" / "python"
_SIMPLER_ROOT = _FRAMEWORK_ROOT / "3rdparty" / "simpler"
_SIMPLER_PYTHON = _SIMPLER_ROOT / "python"
_SIMPLER_SCRIPTS = _SIMPLER_ROOT / "examples" / "scripts"

for path in [_PYPTO_ROOT, _SIMPLER_PYTHON, _SIMPLER_SCRIPTS]:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


class StandaloneRunner:
    """Standalone runner for testing completed orchestration code."""

    def __init__(self):
        """Initialize standalone runner."""
        pass

    def run_completed_test(
        self,
        test_dir: str,
        platform: str = "a2a3sim",
        device_id: int = 0,
    ) -> None:
        """Run a test with manually completed orchestration code.

        Args:
            test_dir: Path to test directory containing:
                      - kernels/orchestration/orch.cpp (your completed orchestration)
                      - kernels/kernel_config.py (generated config)
                      - kernels/golden.py (golden reference)
            platform: Target platform ("a2a3sim" or "a2a3")
            device_id: Device ID for hardware platform

        Raises:
            ValueError: If required files are missing
            Exception: If test execution fails
        """
        test_path = Path(test_dir)
        if not test_path.exists():
            raise ValueError(f"Test directory not found: {test_dir}")

        kernels_dir = test_path / "kernels"
        golden_path = kernels_dir / "golden.py"
        kernel_config = kernels_dir / "kernel_config.py"
        orch_path = kernels_dir / "orchestration" / "orch.cpp"

        # Validate required files
        missing = []
        if not kernels_dir.exists():
            missing.append(str(kernels_dir))
        if not golden_path.exists():
            missing.append(str(golden_path))
        if not kernel_config.exists():
            missing.append(str(kernel_config))
        if not orch_path.exists():
            missing.append(str(orch_path))

        if missing:
            raise ValueError(
                f"Required files missing in {test_dir}:\n  " + "\n  ".join(missing)
            )

        # Check if orchestration contains TODO marker (not completed)
        orch_content = orch_path.read_text()
        if "TODO: Add your task creation code here" in orch_content:
            print("\n⚠️  WARNING: Orchestration file contains TODO marker!")
            print(f"⚠️  Please complete the task creation in: {orch_path}")
            print("⚠️  The test may fail without proper task implementation.\n")

        # Execute via CodeRunner
        from code_runner import CodeRunner

        print(f"\n{'='*60}")
        print(f"Running test: {test_path.name}")
        print(f"Platform: {platform}")
        print(f"Device ID: {device_id}")
        print(f"{'='*60}\n")

        runner = CodeRunner(
            kernels_dir=str(kernels_dir),
            golden_path=str(golden_path),
            platform=platform,
            device_id=device_id,
        )

        try:
            runner.run()
            print(f"\n{'='*60}")
            print("✓ Test PASSED")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"\n{'='*60}")
            print("✗ Test FAILED")
            print(f"Error: {e}")
            print(f"{'='*60}\n")
            raise


def main():
    """Command-line interface for standalone runner."""
    parser = argparse.ArgumentParser(
        description="Standalone runner for testing completed orchestration code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a completed test on simulator
  %(prog)s --run --test-dir ./build/outputs/output_20260202_012129/tile_add_128x128

  # Run on hardware device
  %(prog)s --run --test-dir ./my_test --platform a2a3 --device-id 0

Typical workflow:
  1. Generate skeleton code using TestRunner with codegen_only=True
  2. Edit kernels/orchestration/orch.cpp to add task creation logic
  3. Use this tool to run the completed test on device
        """,
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the test (default action)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Path to test directory (contains kernels/ subdirectory)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="a2a3sim",
        choices=["a2a3sim", "a2a3"],
        help="Target platform (default: a2a3sim)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID for hardware platform (default: 0)",
    )

    args = parser.parse_args()

    # Default action is to run
    if not args.run:
        args.run = True

    if args.run:
        runner = StandaloneRunner()
        try:
            runner.run_completed_test(
                args.test_dir,
                platform=args.platform,
                device_id=args.device_id,
            )
        except Exception as e:
            print(f"\n✗ Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
