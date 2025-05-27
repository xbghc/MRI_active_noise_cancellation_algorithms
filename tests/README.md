# 测试文件夹

本文件夹包含项目的所有测试代码。

## 测试文件

- `test_hyc_data_loader.py` - HycDataLoader类的测试
- `test_comparison.py` - Comparison类的测试  
- `test_mat.py` - MAT文件处理功能的测试
- `run_all_tests.py` - 主测试运行器

## 运行测试

### 运行单个测试

```bash
# 测试数据加载器
python tests/test_hyc_data_loader.py

# 测试比较功能
python tests/test_comparison.py

# 测试MAT文件处理
python tests/test_mat.py
```

### 运行所有测试

```bash
python tests/run_all_tests.py
```

## 测试说明

- 所有测试文件都使用绝对导入
- 测试代码与功能代码完全分离
- 每个测试文件都可以独立运行
- 测试需要项目根目录下有相应的数据文件 