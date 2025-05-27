"""
Comparison 测试文件
"""

import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.hyc_data_loader import HycDataLoader
from models.traditional.editer import EDITER
from models.traditional.yanglei import yanglei
from utils.comparison import Comparison


def test_comparison():
    """测试 Comparison 类的功能"""
    print("=== 测试 Comparison 功能 ===")

    try:
        # 加载噪声数据用于训练
        noise_loader = HycDataLoader("datasets/HYC", set_id=4, data_type="noise")
        primary_train_exp, external_train_exp = noise_loader[0]  # 第一个实验
        # primary_train_exp 形状: (views, views2, samples)
        # external_train_exp 形状: (n_coils, views, views2, samples)

        # 选择第一个views2切片用于训练
        primary_train = primary_train_exp[:, 0, :]  # (views, samples)
        external_train = external_train_exp[:, :, 0, :]  # (n_coils, views, samples)

        editer = EDITER(W=32)
        editer.train(primary_train, external_train)

        # 加载扫描数据用于测试
        scan_loader = HycDataLoader("datasets/HYC", set_id=4, data_type="scan")

        comparison = Comparison()
        # 添加测试数据（使用前4个实验，每个实验选择第一个views2切片）
        for i in range(min(4, len(scan_loader))):
            prim_exp, ext_exp = scan_loader[i]
            # prim_exp 形状: (views, views2, samples), ext_exp 形状: (n_coils, views, views2, samples)

            # 选择第一个views2切片
            prim = prim_exp[:, 0, :]  # (views, samples)
            ext = ext_exp[:, :, 0, :]  # (n_coils, views, samples)

            comparison.add_data(prim, ext)

        comparison.add_algorithm(editer.cancel_noise, "EDITER")
        comparison.add_algorithm(yanglei, "yanglei")

        print("比较测试设置完成")
        print(f"数据集数量: {len(comparison._datas)}")
        print(f"算法数量: {len(comparison._algorithms)}")

        # 可选：显示图像（需要GUI环境）
        # comparison.show_images()
        # comparison.show_diff()

    except Exception as e:
        print(f"比较测试失败: {e}")


if __name__ == "__main__":
    test_comparison()
