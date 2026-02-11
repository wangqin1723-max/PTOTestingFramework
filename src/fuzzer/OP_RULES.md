# PyPTO 算子组合规则 (Op Combination Rules)

本文档定义了 PyPTO IR 中所有支持的算子及其组合规则，用于指导 fuzzer 生成合法的算子组合。

## 0. 形状对齐约束 (Shape Alignment Constraints)

### 0.1 32字节对齐规则

**重要**: 所有 tensor 创建和 reshape 操作必须满足 32 字节对齐约束。

**规则**:
- 形状的**尾轴**(最后一个维度，即列数)必须满足以下条件之一：
  1. 尾轴 = 1, 或者
  2. (尾轴 × sizeof(datatype)) % 32 == 0

**数据类型大小**:
- FP32: 4 字节
- FP16: 2 字节
- INT32: 4 字节
- INT8: 1 字节

**FP32 类型的有效尾轴值**:
- 尾轴 = 1 (总是有效)
- 尾轴 % 8 == 0 (因为 8 × 4 = 32)
- 有效值: 1, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, ...

**示例 (FP32)**:
```python
# ✓ 有效的形状
pl.tensor.create([128, 1], pl.FP32)      # 尾轴=1
pl.tensor.create([128, 8], pl.FP32)      # 8*4=32, 对齐
pl.tensor.create([128, 16], pl.FP32)     # 16*4=64, 对齐
pl.tensor.create([128, 32], pl.FP32)     # 32*4=128, 对齐
pl.tensor.create([128, 64], pl.FP32)     # 64*4=256, 对齐
pl.tensor.create([128, 128], pl.FP32)    # 128*4=512, 对齐

# ✗ 无效的形状
pl.tensor.create([128, 3], pl.FP32)      # 3*4=12, 不对齐
pl.tensor.create([128, 5], pl.FP32)      # 5*4=20, 不对齐
pl.tensor.create([128, 7], pl.FP32)      # 7*4=28, 不对齐
pl.tensor.create([128, 10], pl.FP32)     # 10*4=40, 不对齐 (40 % 32 = 8)
```

**Reshape 约束**:
```python
# 示例: reshape 操作也必须满足对齐约束
tile_tmp = pl.create_tile([8, 1], dtype=pl.FP32, target_memory=1)  # ✓ 尾轴=1
tile_reshaped = pl.reshape(tile_tmp, [1, 8])                        # ✓ 尾轴=8, 8*4=32

# ✗ 错误示例
tile_bad = pl.reshape(tile_tmp, [2, 4])   # ✗ 尾轴=4, 4*4=16, 不对齐
```

### 0.2 Fuzzer 中的对齐验证

Fuzzer 框架提供以下工具函数：

```python
from src.fuzzer.src.fuzzer import is_shape_aligned, generate_aligned_shape, get_aligned_shapes

# 检查形状是否对齐
is_valid = is_shape_aligned((128, 64), dtype="FP32")  # True
is_valid = is_shape_aligned((128, 5), dtype="FP32")   # False

# 生成随机的对齐形状
shape = generate_aligned_shape(rng, dtype="FP32", max_size=256)

# 获取所有常用的对齐形状列表
all_shapes = get_aligned_shapes(dtype="FP32")
```

**Fuzzer 自动处理**:
- `KernelGenerator.generate_kernel()` 会自动验证并修正输入/输出形状
- `OrchestratorGenerator` 会验证所有临时 tensor 的形状
- 如果检测到不对齐的形状，会打印警告并自动生成对齐的形状

## 1. 算子分类 (Operator Categories)

### 1.1 Block Memory Operations (内存操作)

| 算子名 | 输入类型 | 输出类型 | 参数 | 约束 |
|--------|----------|----------|------|------|
| `block.load` | `tensor` | `tile` | `offsets: [int, int]`, `shapes: [int, int]`, `target_memory: int` | target_memory ∈ {1, 2} (UB, L1) |
| `block.store` | `tile` | `tensor` | `offsets: [int, int]`, `shapes: [int, int]`, `output_tensor: tensor` | - |
| `block.l0c_store` | `tile` | `tensor` | `offsets: [int, int]`, `shapes: [int, int]`, `output_tensor: tensor` | - |
| `block.move` | `tile` | `tile` | `target_memory: int`, `transpose: bool` | target_memory ∈ {1, 2, 3, 4} |
| `block.create_tile` | - | `tile` | `shape: [int, int]`, `dtype: DataType`, `target_memory: int` | - |
| `block.full` | - | `tile` | `shape: [int, int]`, `dtype: DataType`, `value: float` | 创建填充值的tile |

### 1.2 Block Element-wise Binary Operations (逐元素二元操作)

| 算子名 | 输入类型 | 输出类型 | 形状约束 | NumPy等价 |
|--------|----------|----------|----------|-----------|
| `block.add` | `tile, tile` | `tile` | 支持广播 | `a + b` |
| `block.sub` | `tile, tile` | `tile` | 支持广播 | `a - b` |
| `block.mul` | `tile, tile` | `tile` | 支持广播 | `a * b` |
| `block.div` | `tile, tile` | `tile` | 支持广播，避免除零 | `a / b` |
| `block.maximum` | `tile, tile` | `tile` | 支持广播 | `np.maximum(a, b)` |
| `block.minimum` | `tile, tile` | `tile` | 支持广播 | `np.minimum(a, b)` |
| `block.cmp` | `tile, tile` | `tile` | 支持广播 | 比较操作，cmp_type: 0=EQ, 1=NE, 2=LT, 3=LE, 4=GT, 5=GE |

### 1.3 Block Scalar Operations (标量操作)

| 算子名 | 输入类型 | 输出类型 | NumPy等价 |
|--------|----------|----------|-----------|
| `block.adds` | `tile, scalar` | `tile` | `a + s` |
| `block.subs` | `tile, scalar` | `tile` | `a - s` |
| `block.muls` | `tile, scalar` | `tile` | `a * s` |
| `block.divs` | `tile, scalar` | `tile` | `a / s` (避免除零) |
| `block.cmps` | `tile, scalar` | `tile` | 比较操作 |

### 1.4 Block Unary Operations (一元操作)

| 算子名 | 输入类型 | 输出类型 | 约束 | NumPy等价 |
|--------|----------|----------|------|-----------|
| `block.neg` | `tile` | `tile` | - | `-a` |
| `block.exp` | `tile` | `tile` | 建议输入范围 [-10, 10] | `np.exp(a)` |
| `block.recip` | `tile` | `tile` | 避免除零 | `1.0 / a` |
| `block.sqrt` | `tile` | `tile` | 输入必须 ≥ 0 | `np.sqrt(a)` |
| `block.rsqrt` | `tile` | `tile` | 输入必须 > 0 | `1.0 / np.sqrt(a)` |
| `block.log` | `tile` | `tile` | 输入必须 > 0 | `np.log(a)` |
| `block.abs` | `tile` | `tile` | - | `np.abs(a)` |
| `block.relu` | `tile` | `tile` | - | `np.maximum(0, a)` |
| `block.cast` | `tile` | `tile` | 参数: `target_dtype: DataType`, `mode: int` | 类型转换 |

### 1.5 Block Matrix Operations (矩阵操作)

| 算子名 | 输入类型 | 输出类型 | 形状约束 | NumPy等价 |
|--------|----------|----------|----------|-----------|
| `block.matmul` | `tile, tile` | `tile` | `[M, K] @ [K, N] -> [M, N]` | `a @ b` |
| `block.matmul_acc` | `tile, tile, tile` | `tile` | `acc + (lhs @ rhs)` | `acc + a @ b` |

### 1.6 Block Row/Column Broadcast Operations (行列广播操作)

**重要**: 这些操作用于处理向量与矩阵的广播运算。

| 算子名 | 输入类型 | 输出类型 | 形状约束 | NumPy等价 |
|--------|----------|----------|----------|-----------|
| `block.row_expand_add` | `tile[M,N], tile[M,1]` | `tile[M,N]` | row_vec广播到每行 | `tile + row_vec` |
| `block.row_expand_sub` | `tile[M,N], tile[M,1]` | `tile[M,N]` | row_vec广播到每行 | `tile - row_vec` |
| `block.row_expand_mul` | `tile[M,N], tile[M,1]` | `tile[M,N]` | row_vec广播到每行 | `tile * row_vec` |
| `block.row_expand_div` | `tile[M,N], tile[M,1]` | `tile[M,N]` | row_vec广播到每行，避免除零 | `tile / row_vec` |
| `block.col_expand` | `tile[M,N], tile[1,N]` | `tile[M,N]` | col_vec广播到每列 | 列向量扩展 |
| `block.col_expand_mul` | `tile[M,N], tile[1,N]` | `tile[M,N]` | col_vec广播到每列 | `tile * col_vec` |
| `block.col_expand_div` | `tile[M,N], tile[1,N]` | `tile[M,N]` | col_vec广播到每列，避免除零 | `tile / col_vec` |
| `block.col_expand_sub` | `tile[M,N], tile[1,N]` | `tile[M,N]` | col_vec广播到每列 | `tile - col_vec` |
| `block.expands` | `tile[M,N], scalar` | `tile[M,N]` | 标量广播到tile形状 | 标量扩展 |

### 1.7 Block Reduction Operations (归约操作)

| 算子名 | 输入类型 | 输出类型 | 参数 | 形状变换 | NumPy等价 |
|--------|----------|----------|------|----------|-----------|
| `block.sum` | `tile` | `tile` | `axis: int`, `keepdim: bool` | axis=1, keepdim=True: [M,N]->[M,1] | `np.sum(a, axis=axis, keepdims=keepdim)` |
| `block.max` | `tile` | `tile` | `axis: int`, `keepdim: bool` | 同上 | `np.max(a, axis=axis, keepdims=keepdim)` |
| `block.min` | `tile` | `tile` | `axis: int`, `keepdim: bool` | 同上 | `np.min(a, axis=axis, keepdims=keepdim)` |
| `block.row_sum` | `tile, tile` | `tile` | 需要临时tile | [M,N] -> [M,1] | `np.sum(a, axis=1, keepdims=True)` |
| `block.row_max` | `tile, tile` | `tile` | 需要临时tile | [M,N] -> [M,1] | `np.max(a, axis=1, keepdims=True)` |
| `block.row_min` | `tile, tile` | `tile` | 需要临时tile | [M,N] -> [M,1] | `np.min(a, axis=1, keepdims=True)` |

### 1.8 Block Transform Operations (变换操作)

| 算子名 | 输入类型 | 输出类型 | 参数 | 形状变换 |
|--------|----------|----------|------|----------|
| `block.reshape` | `tile` | `tile` | `shape: [int, int]` | 重塑形状 |
| `block.transpose` | `tile` | `tile` | `axis1: int`, `axis2: int` | 交换维度 |
| `block.view` | `tile` | `tile` | `shape: [int, int]`, `offset: [int, int]` | 创建视图 |

### 1.9 Tensor-level Operations (Tensor级别操作)

| 算子名 | 输入类型 | 输出类型 | 说明 |
|--------|----------|----------|------|
| `tensor.create` | - | `tensor` | 创建tensor |
| `tensor.view` | `tensor` | `tensor` | 创建tensor视图 |
| `tensor.matmul` | `tensor, tensor` | `tensor` | tensor级矩阵乘法 |
| `tensor.mul` | `tensor, tensor/scalar` | `tensor` | tensor级乘法 |
| `tensor.add` | `tensor, tensor/scalar` | `tensor` | tensor级加法 |
| `tensor.sub` | `tensor, tensor/scalar` | `tensor` | tensor级减法 |
| `tensor.div` | `tensor, tensor/scalar` | `tensor` | tensor级除法 |

## 2. 算子组合规则 (Combination Rules)

### 2.1 基本组合规则

1. **类型匹配**: 操作符的输入类型必须匹配
   - `tile` 操作符接受 `tile` 类型
   - `tensor` 操作符接受 `tensor` 类型
   - 不能混用

2. **形状兼容性**:
   - 二元操作支持广播：`[M,N] op [M,N]`, `[M,N] op [M,1]`, `[M,N] op [1,N]`
   - Row expand 操作: 第二个输入必须是 `[M,1]` 形状
   - Col expand 操作: 第二个输入必须是 `[1,N]` 形状
   - Matmul: `[M,K] @ [K,N] -> [M,N]`

3. **数据约束**:
   - **避免除零**: `div`, `divs`, `recip`, `row_expand_div`, `col_expand_div`
     - 确保分母绝对值 ≥ 0.01
   - **正值约束**: `sqrt`, `rsqrt`, `log`
     - 确保输入 > 0 或使用 `abs(x) + 1e-6`
   - **范围约束**: `exp`
     - 建议输入范围 [-10, 10] 避免溢出

### 2.2 常见算子组合模式

#### 模式1: Softmax 组件
```python
# Step 1: Row max reduction
max_vals = pl.row_max(tile, tmp_tile)  # [M,N] -> [M,1]

# Step 2: Subtract max (数值稳定性)
centered = pl.row_expand_sub(tile, max_vals)  # [M,N] - [M,1] -> [M,N]

# Step 3: Exponential
exp_vals = pl.exp(centered)  # [M,N] -> [M,N]

# Step 4: Row sum
sum_vals = pl.row_sum(exp_vals, tmp_tile)  # [M,N] -> [M,1]

# Step 5: Normalize
output = pl.row_expand_div(exp_vals, sum_vals)  # [M,N] / [M,1] -> [M,N]
```

#### 模式2: Layer Normalization 组件
```python
# Step 1: Row mean (使用 sum + divs)
row_sum = pl.row_sum(tile, tmp_tile)  # [M,N] -> [M,1]
row_mean = pl.divs(row_sum, N)  # [M,1] / scalar -> [M,1]

# Step 2: Subtract mean
centered = pl.row_expand_sub(tile, row_mean)  # [M,N] - [M,1] -> [M,N]

# Step 3: Squared
squared = pl.mul(centered, centered)  # [M,N] * [M,N] -> [M,N]

# Step 4: Variance
var_sum = pl.row_sum(squared, tmp_tile)  # [M,N] -> [M,1]
variance = pl.divs(var_sum, N)  # [M,1] / scalar -> [M,1]

# Step 5: Inverse std
inv_std = pl.rsqrt(variance)  # [M,1] -> [M,1]

# Step 6: Normalize
output = pl.row_expand_mul(centered, inv_std)  # [M,N] * [M,1] -> [M,N]
```

#### 模式3: GELU 近似
```python
# GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
# 简化版本: 使用 sigmoid 近似
# GELU(x) ≈ x * sigmoid(1.702 * x)

# Step 1: Scale
scaled = pl.muls(tile, 1.702)  # [M,N] * scalar -> [M,N]

# Step 2: Sigmoid approximation (使用 exp)
neg_scaled = pl.neg(scaled)  # -[M,N]
exp_neg = pl.exp(neg_scaled)  # exp(-scaled)
one_plus_exp = pl.adds(exp_neg, 1.0)  # 1 + exp(-scaled)
sigmoid = pl.recip(one_plus_exp)  # 1 / (1 + exp(-scaled))

# Step 3: Multiply
output = pl.mul(tile, sigmoid)  # [M,N] * [M,N] -> [M,N]
```

#### 模式4: ReLU 及变体
```python
# ReLU
output = pl.relu(tile)

# LeakyReLU (alpha=0.01)
neg_part = pl.muls(tile, 0.01)  # 负半部分
output = pl.maximum(tile, neg_part)  # max(x, 0.01*x)

# ELU (alpha=1.0) - 简化版
# ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
zeros = pl.expands(tile, 0.0)
pos_mask = pl.maximum(tile, zeros)  # 正半部分
exp_x = pl.exp(tile)  # exp(x)
exp_minus_1 = pl.subs(exp_x, 1.0)  # exp(x) - 1
# 需要 select 操作来完整实现
```

### 2.3 禁止的算子组合

1. **类型混用**:
   ```python
   # ✗ 错误: 不能直接对 tensor 使用 block 操作
   tile_result = pl.add(tensor_a, tensor_b)

   # ✓ 正确: 先 load 到 tile
   tile_a = pl.load(tensor_a, offsets=[0, 0], shapes=[M, N])
   tile_b = pl.load(tensor_b, offsets=[0, 0], shapes=[M, N])
   tile_result = pl.add(tile_a, tile_b)
   ```

2. **形状不匹配**:
   ```python
   # ✗ 错误: row_expand 操作需要 [M,1] 形状
   tile_a = [128, 128]
   tile_b = [128, 64]  # 错误形状
   result = pl.row_expand_div(tile_a, tile_b)

   # ✓ 正确: 使用 reshape 或正确的 load 形状
   tile_b = pl.load(b, offsets=[0, 0], shapes=[128, 1])  # [128,1]
   result = pl.row_expand_div(tile_a, tile_b)
   ```

3. **未处理的数值约束**:
   ```python
   # ✗ 错误: 可能除零
   result = pl.div(tile_a, tile_b)

   # ✓ 正确: 确保分母不为零
   tile_b_safe = pl.maximum(tile_b, pl.expands(tile_b, 0.01))
   result = pl.div(tile_a, tile_b_safe)
   ```

## 3. Fuzzer 生成策略

### 3.1 操作符选择权重

基于实际硬件支持和测试价值，建议权重:

- **高频操作** (权重 10): `add`, `mul`, `sub`, `maximum`, `adds`, `muls`
- **中频操作** (权重 5): `div`, `sqrt`, `exp`, `row_expand_*`, `matmul`
- **低频操作** (权重 2): `rsqrt`, `log`, `recip`, `transpose`, `reshape`
- **特殊操作** (权重 1): `cast`, `cmp`, reduction 操作

### 3.2 形状生成策略

支持的形状规格:
- **标准方形**: 32x32, 64x64, 96x96, 128x128, 256x256
- **长方形**: 64x128, 128x64, 80x96, 96x80, 128x256
- **向量形状**: Nx1, 1xN (用于 row/col expand)

### 3.3 操作链生成规则

1. **长度范围**: 3-10 个操作
2. **变量重用**: 每个中间结果至少使用一次
3. **输入使用**: 所有输入必须至少被使用一次
4. **类型一致性**: 操作链内保持 tile 类型
5. **形状追踪**: 追踪每个变量的形状以确保兼容性

### 3.4 测试用例模板

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel_func(self, a: pl.Tensor[[M, N], pl.FP32],
                      b: pl.Tensor[[M, 1], pl.FP32]) -> pl.Tensor[[M, N], pl.FP32]:
    # Load tiles
    tile_a = pl.load(a, offsets=[0, 0], shapes=[M, N])
    tile_b = pl.load(b, offsets=[0, 0], shapes=[M, 1])

    # Operation chain (fuzzer generated)
    tmp_0 = pl.row_expand_div(tile_a, tile_b)
    tmp_1 = pl.sqrt(tmp_0)
    tmp_2 = pl.muls(tmp_1, 2.0)
    # ... more operations

    return tmp_final
```

## 4. 参考示例

完整示例见: [tests/test_cases/test_expand.py](../../tests/test_cases/test_expand.py)

主要展示了:
- 如何使用 `row_expand_div` 操作
- 如何处理不同形状的输入
- 如何编写 `compute_expected` 参考实现
