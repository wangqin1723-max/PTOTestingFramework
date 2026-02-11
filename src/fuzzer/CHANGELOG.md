# Fuzzer 框架更新日志

## 2026-02-11 - 输入形状一致性修复

### Bug 修复

**修复形状不一致导致的 NumPy 广播错误**

**问题**: 测试用例配置中使用了不同维度的输入形状，导致 NumPy 参考实现中出现广播错误
```python
# ✗ 错误：不同维度的输入
"input_shapes_list": [
    [(128, 128), (64, 64)],           # kernel_0: 不同维度
    [(128, 128), (128, 128), (96, 96)],  # kernel_1: 混合维度
]

# NumPy 计算时报错:
# ValueError: operands could not be broadcast together with shapes (128,128) (96,96)
```

**修复后**:
```python
# ✓ 正确：所有输入使用相同维度
"input_shapes_list": [
    [(128, 128), (128, 128)],           # kernel_0: 统一维度
    [(128, 128), (128, 128), (128, 128)],  # kernel_1: 统一维度
]
```

**影响范围**:
- **src/fuzzer/example_multi_kernel.py**:
  - `fuzz_sequential_simple`: 所有输入改为 128x128
  - `fuzz_branching_parallel`: 所有输入改为 128x128
  - `fuzz_branching_wide`: 所有输入改为 128x128

**根本原因**:
- 当内核中有操作涉及不同形状的输入时（如 96x96 和 128x128），会导致 NumPy 广播失败
- 虽然 PyPTO IR 代码生成时使用了正确的 load 形状，但运算过程中仍会出现形状不匹配

**设计决策**:
- 简化测试用例配置，统一使用相同形状的输入
- 避免在计算过程中处理复杂的形状变换逻辑
- 确保 NumPy 参考实现和 PyPTO IR 代码行为一致

**症状**: `ValueError: operands could not be broadcast together with shapes (128,128) (96,96)`

---

## 2026-02-11 - NumPy 嵌套函数修复

### Bug 修复

**修复 compute_expected 中嵌套函数的 self 参数问题**

**问题**: 生成的 NumPy 参考实现函数包含了错误的 `self` 参数
```python
def compute_expected(self, tensors, params=None):
    def _numpy_kernel_0(self, a, b):  # ✗ 错误：嵌套函数不应该有 self
        ...
    result_0 = self._numpy_kernel_0(...)  # ✗ 错误调用方式
```

**修复后**:
```python
def compute_expected(self, tensors, params=None):
    def _numpy_kernel_0(a, b):  # ✓ 正确：嵌套函数不需要 self
        ...
    result_0 = _numpy_kernel_0(...)  # ✓ 正确：直接调用
```

**影响范围**:
- **src/fuzzer/src/multi_kernel_test_generator.py**:
  - `_generate_numpy_reference()`: 移除嵌套函数的 `self` 参数（第281行）
  - `_generate_test_class()`: 所有调用改为直接调用而不使用 `self.`（第532、546、574、599行）

**症状**: `NameError: name 'self' is not defined`

---

## 2026-02-11 - 形状大小限制

### 性能优化

**限制最大形状尺寸**: 避免内存溢出，将最大形状从 256x256 限制到 128x128

**变更内容**:
1. **fuzzer.py**:
   - `get_aligned_shapes()`: 添加 `max_size` 参数，默认 128
   - `generate_aligned_shape()`: 默认 `max_size` 改为 128
   - 常用行数列表从 `[32, 64, 80, 96, 128, 160, 192, 224, 256]` 改为 `[32, 64, 80, 96, 128]`

2. **example_multi_kernel.py**:
   - 所有 256x256 形状改为 96x96
   - 示例配置使用更小、更安全的形状组合

**原因**:
- 避免超过硬件内存限制
- 提高测试执行速度
- 减少内存分配失败的风险

**影响**:
- 生成的测试用例形状范围: 32x32 到 128x128
- 对齐的列数: 1, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128

---

## 2026-02-11 - Orchestrator 模式修正

### 架构变更 (Breaking Change)

**Orchestrator 不再显式创建 tensor**: 修正 Orchestration 函数以匹配 PyPTO 框架的正确模式

**变更前** (错误):
```python
@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(self, a: ..., b: ...) -> ...:
    # ✗ 错误: 不应该显式创建 tensor
    tmp_0 = pl.tensor.create([128, 128], pl.FP32)
    tmp_1 = pl.tensor.create([128, 128], pl.FP32)

    tmp_0 = self.kernel_0(a, b, tmp_0)
    tmp_1 = self.kernel_1(tmp_0, b, tmp_1)
    return tmp_1
```

**变更后** (正确):
```python
@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(self, a: ..., b: ...) -> ...:
    # ✓ 正确: 框架自动管理输出 tensor
    result_0 = self.kernel_0(a, b)
    result_1 = self.kernel_1(result_0, b)
    return result_1
```

**关键区别**:
1. **Orchestration 函数不创建 tensor**: 移除所有 `pl.tensor.create()` 调用
2. **调用 InCore 函数时只传入输入参数**: 不需要传递输出 tensor
3. **框架自动管理输出**: PyPTO 框架会自动分配和管理 InCore 函数的输出 tensor

**InCore 函数签名保持不变**:
```python
@pl.function(type=pl.FunctionType.InCore)
def kernel_0(self, a: ..., b: ..., output: ...) -> ...:
    # InCore 函数仍然需要 output 参数
    tile_a = pl.load(a, ...)
    result = pl.store(tile_result, ..., output_tensor=output)
    return result
```

### 影响范围

- **src/fuzzer/src/orchestrator_generator.py**:
  - `generate_sequential()`: 移除 tensor 创建，简化 kernel 调用
  - `generate_branching()`: 已经正确，无需修改
  - `generate_mixed()`: 已经正确，无需修改
  - `generate_merge_kernel()`: 移除对齐验证（仍保留 output 参数）

### 参考

- [tests/test_cases/test_expand.py](../../tests/test_cases/test_expand.py): Orchestration 模式参考
- [tests/test_cases/test_matmul.py](../../tests/test_cases/test_matmul.py): Orchestration 模式参考

---

## 2026-02-11 - 形状对齐约束和验证

### 新增功能

1. **32字节对齐约束** (fuzzer.py)
   - 添加 `is_shape_aligned()` 函数验证形状是否满足32字节对齐
   - 添加 `get_aligned_shapes()` 函数获取所有对齐的常用形状
   - 添加 `generate_aligned_shape()` 函数随机生成对齐的形状
   - 支持多种数据类型: FP32, FP16, INT32, INT8

2. **自动形状验证** (kernel_generator.py)
   - `generate_kernel()` 自动验证输入输出形状
   - 检测到不对齐的形状时自动生成对齐的替代形状
   - 打印警告信息提示形状不对齐

3. **Orchestrator 形状验证** (orchestrator_generator.py)
   - 在创建临时 tensor 时验证形状对齐
   - 在 merge_kernel 生成时验证形状对齐
   - 打印警告信息提示不对齐的形状

4. **文档更新** (OP_RULES.md)
   - 新增第 0 节: 形状对齐约束
   - 详细说明32字节对齐规则
   - 提供对齐和不对齐的形状示例
   - 说明 Fuzzer 中的对齐验证工具

### 对齐规则

**核心约束**:
- 形状的尾轴(列数)必须满足: `cols == 1` 或 `(cols * sizeof(dtype)) % 32 == 0`

**FP32 类型的有效尾轴值**:
- 1, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, ..., 128, ...

**示例**:
```python
# ✓ 有效
(128, 1)   # 尾轴=1
(128, 8)   # 8*4=32
(128, 64)  # 64*4=256
(128, 128) # 128*4=512

# ✗ 无效
(128, 3)   # 3*4=12, 不对齐
(128, 5)   # 5*4=20, 不对齐
(128, 10)  # 10*4=40, 不对齐
```

### 影响范围

- **src/fuzzer/src/fuzzer.py**: 新增对齐验证工具函数
- **src/fuzzer/src/kernel_generator.py**: 导入并使用对齐验证
- **src/fuzzer/src/orchestrator_generator.py**: 导入并使用对齐验证
- **src/fuzzer/OP_RULES.md**: 新增第 0 节文档

### 向后兼容性

- 现有代码如果使用不对齐的形状，会自动修正并打印警告
- 不会导致生成失败，而是自动选择最接近的对齐形状
- 建议手动检查生成的代码，确保形状符合预期

---

## 2026-02-11 - API 简化和算子扩展

### API 变更 (Breaking Change)

**简化 PyPTO API 调用**: 将 `pl.op.block.xxx` 简化为 `pl.xxx`

**变更前**:
```python
tile_a = pl.op.block.load(a, 0, 0, 128, 128)
tmp_0 = pl.op.block.add(tile_a, tile_b)
result = pl.op.block.relu(tmp_0)
```

**变更后**:
```python
tile_a = pl.load(a, offsets=[0, 0], shapes=[128, 128])
tmp_0 = pl.add(tile_a, tile_b)
result = pl.relu(tmp_0)
```

**影响范围**:
- `kernel_generator.py`: 内核代码生成
- `multi_kernel_test_generator.py`: 测试类代码生成
- `orchestrator_generator.py`: 合并内核生成
- `OP_RULES.md`: 文档示例
- `README.md`: 文档示例

### 新增功能

1. **扩展算子支持** (fuzzer.py)
   - 新增一元算子: `log`, `abs`, `relu`
   - 新增二元算子: `minimum`
   - 新增高级算子组:
     - Row expand 系列: `row_expand_add`, `row_expand_sub`, `row_expand_mul`, `row_expand_div`
     - Matrix 算子: `matmul`

2. **高级算子开关**
   - 添加 `enable_advanced_ops` 参数到所有生成器类
   - 基础模式: 使用标准算子 (add, mul, sqrt, exp等)
   - 高级模式: 额外包含 row_expand 和 matmul 算子

3. **算子规则文档** ([OP_RULES.md](OP_RULES.md))
   - 完整的算子分类和定义
   - 每个算子的形状约束说明
   - 常见算子组合模式 (Softmax, LayerNorm, GELU, ReLU变体等)
   - 禁止的算子组合和约束处理
   - Fuzzer 生成策略建议

### 修改文件

1. **src/fuzzer.py**
   - 扩展 `BLOCK_UNARY_OPS`: 新增 log, abs, relu
   - 扩展 `BLOCK_BINARY_OPS`: 新增 minimum
   - 新增 `BLOCK_ROW_EXPAND_OPS`: row_expand_* 系列
   - 新增 `BLOCK_MATRIX_OPS`: matmul
   - 添加 `enable_advanced_ops` 参数
   - 简化 row_expand 操作的输入类型定义

2. **src/kernel_generator.py**
   - 添加 `enable_advanced_ops` 参数支持
   - 传递高级算子开关到 OpFuzzer

3. **src/multi_kernel_test_generator.py**
   - 添加 `enable_advanced_ops` 参数支持
   - 更新 `_get_numpy_operation` 方法支持所有新算子:
     - log, abs, relu, minimum
     - row_expand_add, row_expand_sub, row_expand_mul, row_expand_div
     - matmul

4. **example_multi_kernel.py**
   - 添加 `--enable-advanced-ops` 命令行参数
   - 在输出中显示是否启用高级算子

5. **README.md**
   - 更新快速开始部分，区分基础和高级示例
   - 添加高级算子使用说明
   - 添加对 OP_RULES.md 的引用
   - 更新算子列表和约束说明

### 使用方法

#### 基础模式 (默认)
```bash
python src/fuzzer/example_multi_kernel.py --num-cases 3
```
使用算子: add, sub, mul, div, maximum, minimum, adds, subs, muls, divs, sqrt, rsqrt, exp, neg, recip, log, abs, relu

#### 高级模式
```bash
python src/fuzzer/example_multi_kernel.py --num-cases 3 --enable-advanced-ops
```
额外包含: row_expand_add, row_expand_sub, row_expand_mul, row_expand_div, matmul

### 算子约束

- **avoid_zero**: div, divs, recip, row_expand_div
- **positive_only**: sqrt, rsqrt, log
- **row_vec_required**: row_expand_* 系列 (第二个输入需要 [M,1] 形状)

### 参考文档

- [OP_RULES.md](OP_RULES.md) - 完整的算子规则和组合模式
- [README.md](README.md) - 框架使用文档
- [tests/test_cases/test_expand.py](../../tests/test_cases/test_expand.py) - row_expand 使用示例
