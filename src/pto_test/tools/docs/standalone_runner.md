# Standalone Runner - User Guide

## Overview

The standalone runner allows you to manually complete orchestration code and test it on the device. This is useful when you need to customize the task graph beyond what can be auto-generated.

## Workflow

### Step 1: Generate Skeleton Code

Use `TestRunner` with `codegen_only=True` to generate skeleton orchestration code:

```python
from pto_test.core.test_case import TestConfig
from pto_test.core.test_runner import TestRunner

# Your test case implementation
test_case = MyTestCase()

# Configure for code generation only
config = TestConfig(
    save_kernels=True,
    save_kernels_dir="./my_output",
    codegen_only=True,  # Don't run, just generate
)

runner = TestRunner(config)
result = runner.run(test_case)
```

This generates:
```
./my_output/
└── my_test_name/
    ├── kernels/
    │   ├── aiv/
    │   │   └── kernel_*.cpp
    │   ├── orchestration/
    │   │   └── orch.cpp          ← Edit this file!
    │   ├── kernel_config.py
    │   └── golden.py
    └── metadata.json
```

### Step 2: Complete the Orchestration

Edit `kernels/orchestration/orch.cpp` and replace the TODO section with your task creation logic.

**Before (generated skeleton):**
```cpp
// ========================================
// TODO: Add your task creation code here
// ========================================
//
// Available device tensors:
//   - dev_a (input)
//   - dev_b (input)
//   - dev_c (output)
//   - element_count (size)
//
// Example task creation:
//   uint64_t args_t0[N];
//   args_t0[0] = reinterpret_cast<uint64_t>(dev_input);
//   args_t0[1] = reinterpret_cast<uint64_t>(dev_output);
//   args_t0[2] = element_count;
//   int t0 = runtime->add_task(args_t0, N, func_id, 1);
//
// Available kernel func_ids:
//   - func_id=0: kernel_add
//
// ========================================
```

**After (completed):**
```cpp
// Create task for c = a + b
uint64_t args_t0[4];
args_t0[0] = reinterpret_cast<uint64_t>(dev_a);
args_t0[1] = reinterpret_cast<uint64_t>(dev_b);
args_t0[2] = reinterpret_cast<uint64_t>(dev_c);
args_t0[3] = element_count;
int t0 = runtime->add_task(args_t0, 4, 0, 1);
(void)t0;  // Suppress unused variable warning
```

### Step 3: Run on Device

Use the standalone runner to execute your completed code:

**Command Line:**
```bash
python -m pto_test.tools.standalone_runner \
  --run \
  --test-dir ./my_output/my_test_name \
  --platform a2a3sim
```

**Python API:**
```python
from pto_test.tools.standalone_runner import StandaloneRunner

runner = StandaloneRunner()
runner.run_completed_test(
    './my_output/my_test_name',
    platform='a2a3sim',
    device_id=0
)
```

## Multi-Kernel Example

For complex tests with multiple kernels and dependencies:

```cpp
// Allocate intermediate tensors (if needed)
size_t BYTES = element_count * sizeof(float);
void* dev_temp = runtime->host_api.device_malloc(BYTES);
if (!dev_temp) {
    std::cerr << "Error: Failed to allocate intermediate tensor\n";
    runtime->host_api.device_free(dev_a);
    runtime->host_api.device_free(dev_b);
    runtime->host_api.device_free(dev_c);
    return -1;
}

// Task 0: temp = a + b
uint64_t args_t0[4];
args_t0[0] = reinterpret_cast<uint64_t>(dev_a);
args_t0[1] = reinterpret_cast<uint64_t>(dev_b);
args_t0[2] = reinterpret_cast<uint64_t>(dev_temp);
args_t0[3] = element_count;
int t0 = runtime->add_task(args_t0, 4, 0, 1);

// Task 1: c = temp * 2
uint64_t args_t1[4];
args_t1[0] = reinterpret_cast<uint64_t>(dev_temp);
union { float f32; uint64_t u64; } scalar;
scalar.f32 = 2.0f;
args_t1[1] = scalar.u64;
args_t1[2] = reinterpret_cast<uint64_t>(dev_c);
args_t1[3] = element_count;
int t1 = runtime->add_task(args_t1, 4, 1, 1);

// Add dependency: t0 → t1
runtime->add_successor(t0, t1);
```

## Command Reference

### Standalone Runner Options

```bash
python -m pto_test.tools.standalone_runner --help
```

Options:
- `--run`: Run the test (default action)
- `--test-dir PATH`: Path to test directory (required)
- `--platform {a2a3sim,a2a3}`: Target platform (default: a2a3sim)
- `--device-id N`: Device ID for hardware (default: 0)

## Tips

1. **Check for TODO markers**: The standalone runner will warn you if your orchestration file still contains the TODO marker.

2. **Use the generated hints**: The skeleton code lists all available tensors and kernel func_ids.

3. **Error handling**: Make sure to add proper cleanup code for any intermediate tensors you allocate.

4. **Testing iteration**: You can edit the orchestration file and re-run the standalone runner multiple times without regenerating the skeleton.

## Complete Example

See [example_workflow.py](../example_workflow.py) for a complete end-to-end example.
