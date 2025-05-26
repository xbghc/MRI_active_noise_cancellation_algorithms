# HycDataLoader 使用说明

## 概述

`HycDataLoader` 是用于加载HYC数据集的数据加载器，已经过重新整理以提供更清晰和一致的接口。

## 主要特性

1. **严格的参数验证**：`data_type` 只能为 `"noise"` 或 `"scan"`
2. **索引访问**：支持通过索引直接访问数据，不区分实验
3. **标准化数据格式**：`self._data` 格式为四维数组 `exp × views2 × views × samples`

## 数据格式

### 主线圈数据

- 形状：`(exp, views2, views, samples)`
- 类型：`complex64`

### 外部线圈数据  

- 形状：`(exp, views2, n_coils, views, samples)`
- 类型：`complex64`

## 基本用法

### 1. 初始化数据加载器

```python
from data.hyc_data_loader import HycDataLoader

# 加载噪声数据
noise_loader = HycDataLoader("datasets/HYC", set_id=4, data_type="noise")

# 加载扫描数据
scan_loader = HycDataLoader("datasets/HYC", set_id=4, data_type="scan")
```

### 2. 索引访问

```python
# 通过线性索引访问数据（索引范围：0 到 views2 * exp - 1）
primary_data, external_data = noise_loader[0]
# primary_data 形状: (views, samples)
# external_data 形状: (n_coils, views, samples)

# 访问不同的 (实验, views2) 组合
primary_data_1, external_data_1 = noise_loader[1]  # 第二个数据
primary_data_24, external_data_24 = noise_loader[24]  # 第二个实验的第一个views2（假设每个实验有24个views2）

# 数据集总长度为 views2 * exp
total_length = len(noise_loader)
print(f"总数据量: {total_length}")
```

### 3. 获取数据信息

```python
info = noise_loader.get_data_info()
print(f"实验数量: {info['n_experiments']}")
print(f"外部线圈数量: {info['n_external_coils']}")
print(f"数据维度: views2={info['views2']}, views={info['views']}, samples={info['samples']}")
```

## 实际应用示例

### 训练算法

```python
# 加载训练数据
noise_loader = HycDataLoader("datasets/HYC", set_id=4, data_type="noise")
train_primary, train_external = noise_loader[0]  # 第一个数据（第一个实验的第一个views2）
# train_primary 形状: (views, samples)
# train_external 形状: (n_coils, views, samples)

# 训练算法
from models.traditional.editer import EDITER
editer = EDITER(W=32)
editer.train(train_primary, train_external)
```

### 测试算法

```python
# 加载测试数据
scan_loader = HycDataLoader("datasets/HYC", set_id=4, data_type="scan")
test_primary, test_external = scan_loader[0]  # 第一个数据（第一个实验的第一个views2）
# test_primary 形状: (views, samples)
# test_external 形状: (n_coils, views, samples)

# 应用算法
cleaned_data = editer.cancel_noise(test_primary, test_external)
```

### 批量处理

```python
# 处理所有数据（所有实验的所有views2）
for i in range(len(noise_loader)):
    primary, external = noise_loader[i]
    # primary 形状: (views, samples)
    # external 形状: (n_coils, views, samples)
    
    # 进行处理...
    cleaned_data = some_algorithm(primary, external)
    
    # 如果需要知道当前是哪个实验和views2
    info = noise_loader.get_data_info()
    views2_per_exp = info['views2']
    exp_idx = i // views2_per_exp
    views2_idx = i % views2_per_exp
    print(f"处理实验{exp_idx}的views2_{views2_idx}")
```

## 错误处理

```python
try:
    # 无效的data_type会抛出ValueError
    loader = HycDataLoader("datasets/HYC", set_id=4, data_type="invalid")
except ValueError as e:
    print(f"参数错误: {e}")

try:
    # 不存在的路径会抛出FileNotFoundError
    loader = HycDataLoader("nonexistent_path", set_id=4, data_type="noise")
except FileNotFoundError as e:
    print(f"路径错误: {e}")
```

## 注意事项

1. **数据类型限制**：`data_type` 参数只接受 `"noise"` 或 `"scan"`
2. **索引范围**：确保索引不超出数据集的实验数量范围
3. **内存使用**：所有数据在初始化时加载到内存中，大数据集可能需要注意内存使用
4. **数据完整性**：如果某个实验的数据文件缺失或损坏，该实验会被跳过
