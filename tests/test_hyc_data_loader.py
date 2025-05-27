"""
HycDataLoader 测试文件
"""

import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.hyc_data_loader import HycDataLoader


def test_hyc_data_loader():
    """测试 HycDataLoader 的 flatten 功能"""
    print("=== 测试 HycDataLoader 的 flatten 功能 ===")

    # 3维模式测试
    print("\n1. 3维模式 (flatten=False):")
    try:
        loader_3d = HycDataLoader(
            "datasets/HYC", set_id=4, data_type="noise", flatten=False
        )
        print(f"数据集长度: {len(loader_3d)}")
        print("数据集信息:")
        info = loader_3d.get_data_info()
        for key, value in info.items():
            print(f"  {key}: {value}")

        # 获取第一个实验的数据
        prim, ext = loader_3d[0]
        print("第一个实验数据形状:")
        print(f"  primary: {prim.shape}")
        print(f"  external: {ext.shape}")

    except Exception as e:
        print(f"3维模式测试失败: {e}")

    # 1维模式测试
    print("\n2. 1维模式 (flatten=True):")
    try:
        loader_1d = HycDataLoader(
            "datasets/HYC", set_id=4, data_type="noise", flatten=True
        )
        print(f"数据集长度: {len(loader_1d)}")
        print("数据集信息:")
        info = loader_1d.get_data_info()
        for key, value in info.items():
            print(f"  {key}: {value}")

        # 获取第一行数据
        prim_row, ext_row = loader_1d[0]
        print("第一行数据形状:")
        print(f"  primary: {prim_row.shape}")
        print(f"  external: {ext_row.shape}")

        # 获取最后一行数据
        last_idx = len(loader_1d) - 1
        prim_last, ext_last = loader_1d[last_idx]
        print("最后一行数据形状:")
        print(f"  primary: {prim_last.shape}")
        print(f"  external: {ext_last.shape}")

    except Exception as e:
        print(f"1维模式测试失败: {e}")


def test_data_consistency():
    """测试两种模式下数据的一致性"""
    print("\n=== 测试数据一致性 ===")

    try:
        # 创建两个加载器
        loader_3d = HycDataLoader(
            "datasets/HYC", set_id=4, data_type="noise", flatten=False
        )
        loader_1d = HycDataLoader(
            "datasets/HYC", set_id=4, data_type="noise", flatten=True
        )

        # 获取3维模式的第一个实验数据
        prim_3d, ext_3d = loader_3d[0]
        views, views2, samples = prim_3d.shape
        n_coils = ext_3d.shape[0]

        print(f"3维模式数据形状: primary={prim_3d.shape}, external={ext_3d.shape}")
        print(f"1维模式总长度: {len(loader_1d)}")
        print(
            f"预期长度: {len(loader_3d)} * {views} * {views2} = {len(loader_3d) * views * views2}"
        )

        # 验证第一行数据
        prim_1d_first, ext_1d_first = loader_1d[0]
        prim_expected = prim_3d[0, 0, :]  # 第一个view, 第一个views2
        ext_expected = ext_3d[:, 0, 0, :]  # 所有线圈的第一个view, 第一个views2

        print(
            f"1维模式第一行: primary={prim_1d_first.shape}, external={ext_1d_first.shape}"
        )
        print(
            f"数据一致性检查: {(prim_1d_first == prim_expected).all() and (ext_1d_first == ext_expected).all()}"
        )

    except Exception as e:
        print(f"一致性测试失败: {e}")


if __name__ == "__main__":
    test_hyc_data_loader()
    test_data_consistency()
