# PTO Testing Framework

A general-purpose integration testing framework designed to perform fuzz testing between different frontends and backends.

## Getting Started

### Clone the Repository

This repository uses git submodules. Clone with the `--recursive` flag to fetch all dependencies:

```bash
git clone --recursive https://github.com/<your-org>/pto-testing-framework.git
```

If you have already cloned the repository without `--recursive`, run:

```bash
git submodule update --init --recursive
```

### Build and Installation

The framework provides a build script that compiles PyPTO and automatically configures your environment:

```bash
# Basic build (RelWithDebInfo by default)
./build_and_install.sh

# Clean build from scratch
./build_and_install.sh --clean

# Debug build
./build_and_install.sh --type Debug

# Install pto-test package in editable mode
./build_and_install.sh --install

# Parallel build with 8 jobs
./build_and_install.sh --jobs 8

# Combine options
./build_and_install.sh --clean --type Release --install
```

**Build Script Options:**
- `-t, --type TYPE` — Build type: Debug, Release, RelWithDebInfo (default: RelWithDebInfo)
- `-c, --clean` — Clean build directory before building
- `-i, --install` — Install pto-test package in editable mode
- `-j, --jobs N` — Number of parallel jobs (default: auto-detect)
- `-h, --help` — Show help message

After building, the environment is **automatically configured**. The script generates and sources `build/setup_env.sh`, setting up `PYTHONPATH` and other variables.

For new terminal sessions, source the environment:

```bash
source build/setup_env.sh
```

## Architecture

The framework serves as an integration layer that decouples frontends and backends, enabling comprehensive end-to-end testing between PyPTO (compiler frontend) and Simpler (runtime backend).

### Components

**Frontend:**
- [pypto](https://github.com/hw-native-sys/pypto) — Python DSL for tensor programming, located at `3rdparty/pypto`

**Backend:**
- [simpler](https://github.com/ChaoWao/simpler) — Runtime execution engine, located at `3rdparty/simpler`

**Testing Framework (pto-test):**
The framework bridges frontend and backend through three core modules:

#### Core Infrastructure (`src/pto_test/core/`)
- **test_case.py** — Base classes for test definitions
  - `PTOTestCase`: Abstract base class for user-defined tests
  - `TensorSpec`: Tensor specification with flexible initialization patterns
  - `TestConfig`: Configuration options (platform, tolerances, save options)
  - `TestResult`: Test execution results and validation metrics
- **test_runner.py** — Test execution engine
  - `TestRunner`: Orchestrates the complete test pipeline
  - `TestSuite`: Batch execution and summary reporting
- **validators.py** — Result validation utilities
  - `ResultValidator`: Compares outputs with configurable tolerances

#### Code Generation Pipeline (`src/pto_test/codegen/`)
- **kernel_generator.py** — PyPTO IR → CCE C++ kernels
  - Runs PyPTO pass pipeline with optimization strategies
  - Generates per-function kernel files
- **orch_generator.py** — Auto-generates orchestration C++ skeleton
- **config_generator.py** — Generates `kernel_config.py` for simpler runtime
- **golden_generator.py** — Generates `golden.py` with reference computations

#### Utilities (`src/pto_test/tools/`)
- **standalone_runner.py** — Tool for testing manually completed orchestration code

## Directory Structure

```
pto-testing-framework/
├── src/pto_test/           # Testing framework source
│   ├── core/               # Core test infrastructure
│   │   ├── test_case.py    # PTOTestCase, TensorSpec, TestConfig
│   │   ├── test_runner.py  # TestRunner, TestSuite
│   │   └── validators.py   # ResultValidator
│   ├── codegen/            # Code generation pipeline
│   │   ├── kernel_generator.py    # PyPTO IR → CCE C++ kernels
│   │   ├── orch_generator.py      # Auto-generate orchestration
│   │   ├── config_generator.py    # Generate kernel_config.py
│   │   └── golden_generator.py    # Generate golden.py
│   └── tools/              # Utilities
│       └── standalone_runner.py   # Manual orchestration testing
├── tests/                  # Test cases
│   ├── conftest.py         # pytest configuration and fixtures
│   └── test_cases/         # Actual test implementations
│       └── test_elementwise.py
├── 3rdparty/               # Git submodules
│   ├── pypto/              # PyPTO frontend
│   └── simpler/            # Simpler runtime backend
├── build/                  # Build artifacts (generated)
│   ├── pypto/              # PyPTO build output
│   ├── setup_env.sh        # Auto-generated environment setup
│   └── outputs/            # Test artifacts (when --save-kernels)
│       └── output_{timestamp}/
│           └── {test_name}/
│               ├── kernels/
│               ├── pass_dump/      # (if --dump-passes)
│               └── metadata.json
├── build_and_install.sh    # Build script for PyPTO
└── pyproject.toml          # Project configuration
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests (simulation platform by default)
pytest tests/ -v

# Run specific test file
pytest tests/test_cases/test_elementwise.py -v

# Run specific test class
pytest tests/test_cases/test_elementwise.py::TestTileAdd -v

# Run with verbose output
pytest tests/ -vv
```

### Saving and Debugging

```bash
# Save generated kernels and artifacts
pytest tests/ -v --save-kernels

# Save to custom directory
pytest tests/ -v --save-kernels --kernels-dir ./my_outputs

# Enable IR pass dumps for debugging compiler transformations
pytest tests/ -v --save-kernels --dump-passes

# Generate code only, skip runtime execution
pytest tests/ -v --codegen-only --save-kernels
```

### Hardware Platform

```bash
# Run on real hardware (requires NPU device)
pytest tests/ -v --platform=a2a3 --device=0

# Run on hardware with saved kernels
pytest tests/ -v --platform=a2a3 --device=0 --save-kernels
```

### Using Different Optimization Strategies

```bash
# Use PTOAS optimization strategy
pytest tests/ -v --strategy=PTOAS

# Combine with other options
pytest tests/ -v --strategy=PTOAS --save-kernels --dump-passes
```

## Writing Test Cases

Define test cases by inheriting from `PTOTestCase` and implementing required methods:

```python
from pto_test.core.test_case import PTOTestCase, TensorSpec, DataType
import pypto.lang as pl

class TestTileAdd(PTOTestCase):
    def get_name(self) -> str:
        """Return a unique name for this test."""
        return "tile_add_128x128"

    def define_tensors(self) -> list[TensorSpec]:
        """Define input and output tensors with initialization."""
        return [
            TensorSpec("a", [128, 128], DataType.FP32, init_value=2.0),
            TensorSpec("b", [128, 128], DataType.FP32, init_value=3.0),
            TensorSpec("c", [128, 128], DataType.FP32, is_output=True),
        ]

    def get_program(self):
        """Define the PyPTO program to test."""
        @pl.program
        class TileAddProgram:
            @pl.function
            def tile_add(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                c: pl.Tensor[[128, 128], pl.FP32],
            ):
                tile_a = pl.op.block.load(a, 0, 0, 128, 128)
                tile_b = pl.op.block.load(b, 0, 0, 128, 128)
                tile_c = pl.op.block.add(tile_a, tile_b)
                pl.op.block.store(tile_c, 0, 0, 128, 128, c)
        return TileAddProgram

    def compute_expected(self, tensors: dict, params=None):
        """Compute expected result using NumPy."""
        tensors["c"][:] = tensors["a"] + tensors["b"]
```

### Tensor Initialization Patterns

`TensorSpec` supports multiple initialization patterns:

```python
# Scalar initialization (broadcast to all elements)
TensorSpec("a", [128, 128], DataType.FP32, init_value=1.0)

# NumPy array initialization
TensorSpec("b", [4, 4], DataType.FP32, init_value=np.eye(4))

# Callable initialization (for random data)
TensorSpec("c", [256, 256], DataType.FP32, init_value=lambda: np.random.randn(256, 256))

# Zero initialization (default)
TensorSpec("output", [128, 128], DataType.FP32, is_output=True)
```

### Parameterized Tests

Use pytest fixtures for shape variations:

```python
import pytest

class TestTileAddParameterized(PTOTestCase):
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols

    def get_name(self) -> str:
        return f"tile_add_{self.rows}x{self.cols}"

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("a", [self.rows, self.cols], DataType.FP32, init_value=1.0),
            TensorSpec("b", [self.rows, self.cols], DataType.FP32, init_value=2.0),
            TensorSpec("c", [self.rows, self.cols], DataType.FP32, is_output=True),
        ]
    # ... rest of implementation

@pytest.mark.parametrize("rows,cols", [(64, 64), (128, 128), (256, 256)])
def test_tile_add_shapes(test_runner, rows, cols):
    test_case = TestTileAddParameterized(rows, cols)
    result = test_runner.run(test_case)
    assert result.passed, f"Test failed: {result.error}"
```

## Configuration Options

pytest command-line options for controlling test execution:

| Option | Default | Description |
|--------|---------|-------------|
| `--platform` | a2a3sim | Target platform (a2a3sim for simulation, a2a3 for hardware) |
| `--device` | 0 | Device ID for hardware tests |
| `--strategy` | Default | PyPTO optimization strategy (Default or PTOAS) |
| `--save-kernels` | False | Save generated kernels and artifacts |
| `--kernels-dir` | build/outputs/output_{timestamp}/ | Custom output directory for kernels |
| `--dump-passes` | False | Dump intermediate IR after each pass |
| `--codegen-only` | False | Only generate code, skip runtime execution |
| `--fuzz-count` | 10 | Number of fuzz test iterations (planned feature) |
| `--fuzz-seed` | random | Random seed for fuzz tests (planned feature) |

**Note:** Fuzz testing infrastructure is available but test cases are not yet implemented. The `--fuzz-count` and `--fuzz-seed` options are reserved for future fuzz testing functionality.

## Advanced Usage

### Saving Generated Code

By default, generated kernels and artifacts are in temporary directories. Use `--save-kernels` to persist them:

```bash
# Save to default location (build/outputs/output_{timestamp}/)
pytest tests/ -v --save-kernels

# Save to custom directory
pytest tests/ -v --save-kernels --kernels-dir ./my_outputs

# Run specific test with saved kernels
pytest tests/test_cases/test_elementwise.py::TestTileAdd -v --save-kernels
```

**Output structure:**
```
build/outputs/output_{timestamp}/
└── {test_name}/
    ├── kernels/
    │   ├── aiv/
    │   │   └── {func_name}.cpp      # Generated kernel code
    │   ├── orchestration/
    │   │   └── orch.cpp             # Orchestration skeleton
    │   ├── kernel_config.py         # Config for simpler runtime
    │   └── golden.py                # Reference computation
    └── metadata.json                # Test metadata
```

### Debugging with Pass Dumps

Dump intermediate IR representations after each compiler pass:

```bash
pytest tests/ -v --save-kernels --dump-passes
```

This creates a `pass_dump/` directory in each test output with IR snapshots at each optimization stage, useful for debugging compiler transformations.

### Code Generation Only

Generate code without executing on the runtime (useful for validating codegen):

```bash
pytest tests/ -v --codegen-only --save-kernels
```

### Using Optimization Strategies

Override the default optimization strategy at runtime:

```bash
# Use PTOAS optimization strategy
pytest tests/ -v --strategy=PTOAS
```

Or override in your test case:

```python
class MyTest(PTOTestCase):
    def get_strategy(self):
        from pypto.ir.pass_manager import OptimizationStrategy
        return OptimizationStrategy.PTOAS
```

### Manual Orchestration Testing (Standalone Runner)

The `standalone_runner.py` tool allows you to manually complete and test orchestration code. This is useful when you need custom task creation logic beyond what the auto-generator provides.

**Typical workflow:**

1. **Generate skeleton code** using `--codegen-only`:
```bash
pytest tests/test_cases/test_elementwise.py::TestTileAdd -v --codegen-only --save-kernels
```

2. **Locate the generated orchestration file**:
```
build/outputs/output_{timestamp}/tile_add_128x128/kernels/orchestration/orch.cpp
```

3. **Edit the orchestration file** to replace the `TODO` section with your custom task creation code:
```cpp
// TODO: Add your task creation code here
// Example:
// auto task = runtime->create_task(...);
// runtime->submit_task(task);
```

4. **Run the completed test** using the standalone runner:
```bash
# Run on simulator
python -m pto_test.tools.standalone_runner \
  --run \
  --test-dir build/outputs/output_20260202_012129/tile_add_128x128 \
  --platform a2a3sim

# Run on hardware
python -m pto_test.tools.standalone_runner \
  --run \
  --test-dir ./my_test \
  --platform a2a3 \
  --device-id 0
```

**Programmatic usage:**
```python
from pto_test.tools.standalone_runner import StandaloneRunner

runner = StandaloneRunner()
runner.run_completed_test(
    test_dir='./my_test',
    platform='a2a3sim',
    device_id=0
)
```

**Tool features:**
- Validates presence of required files (kernel_config.py, golden.py, orch.cpp)
- Warns if orchestration still contains TODO markers
- Executes via simpler's CodeRunner
- Provides clear pass/fail feedback

## Supported Operations

### Elementwise Binary Operations
- `add`, `sub`, `mul`, `div`

### Scalar Operations
- `adds`, `subs`, `muls`, `divs`

### Unary Operations
- `exp`, `sqrt`, `neg`, `abs`

## Environment Setup

The `build_and_install.sh` script automatically configures your environment. For new terminal sessions or manual setup:

```bash
# Source the auto-generated environment script
source build/setup_env.sh
```

This sets up:
- `PYTHONPATH` — Includes PyPTO, Simpler, and pto-test packages
- `PYPTO_DIR` — Path to PyPTO source
- `SIMPLER_DIR` — Path to Simpler source

**Verify your setup:**

```bash
# Check environment variables
echo $PYTHONPATH
echo $PYPTO_DIR

# Test imports
python -c "import pypto; print(f'PyPTO version: {pypto.__version__}')"
python -c "import simpler; print('Simpler imported successfully')"
python -c "import pto_test; print('pto-test imported successfully')"
```

**Manual PYTHONPATH (if needed):**

```bash
export PYTHONPATH="${PWD}/3rdparty/pypto/python:${PWD}/3rdparty/simpler/python:${PWD}/src:${PYTHONPATH}"
```

## Test Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Test Case Definition                    │
│    (TensorSpec + build_ir + compute_expected)                │
└───────────────────────────┬─────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐ ┌───────────────────┐ ┌─────────────────┐
│ 1. Build IR     │ │ 2. Generate Kernel│ │ 3. Prepare Data │
│ (PyPTO)         │ │ (KernelGenerator) │ │ (NumPy arrays)  │
└────────┬────────┘ └─────────┬─────────┘ └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │ 4. Compile Components  │
                 │ - Kernel → .o/.text    │
                 │ - Orchestration → .so  │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ 5. Execute on Simpler  │
                 │ - bind_host_binary()   │
                 │ - register_kernel()    │
                 │ - launch_runtime()     │
                 └───────────┬────────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ 6. Validate Results    │
                 │ - Compare with NumPy   │
                 │ - Return TestResult    │
                 └────────────────────────┘
```

## Troubleshooting

### Submodules Not Initialized

**Problem:** `ModuleNotFoundError` for `pypto` or `simpler`, or empty `3rdparty/` directories.

**Solution:**
```bash
git submodule update --init --recursive
```

### Import Errors After Build

**Problem:** `ImportError: No module named 'pypto'` or similar.

**Solution:**
```bash
source build/setup_env.sh
```

### Build Failures

**Problem:** CMake configuration errors or compilation failures.

**Solutions:**
```bash
# Try a clean build
./build_and_install.sh --clean

# Check Python installation
python3 --version
python3 -c "import sys; print(sys.prefix)"

# Ensure nanobind is installed
pip install nanobind
```

### nanobind Not Found

**Problem:** CMake error about missing `nanobind_DIR`.

**Solution:**
```bash
pip install nanobind
```

The build script will automatically install nanobind if missing, but manual installation may be needed in some environments.

### Tests Fail to Run

**Problem:** pytest doesn't discover tests or fixtures are missing.

**Solutions:**
```bash
# Ensure you're in the project root
cd /path/to/pto-testing-framework

# Check conftest.py is present
ls tests/conftest.py

# Run with verbose output
pytest tests/ -vv
```

### Hardware Tests Not Running

**Problem:** Tests marked with `@pytest.mark.hardware` are skipped.

**Solution:**
```bash
# Use the hardware platform
pytest tests/ -v --platform=a2a3 --device=0
```

Simulation tests run by default (`--platform=a2a3sim`).

## License

Apache-2.0
