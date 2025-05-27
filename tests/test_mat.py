"""
Mat 模块测试文件
"""

import os
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import scipy.io as sio

from models.traditional.editer import EDITER
from utils.mat import load_mat_data
from utils.mrd import reconImagesByFFT


def test_mat_processing():
    """测试 MAT 文件处理功能"""
    print("=== 测试 MAT 文件处理功能 ===")

    try:
        file_path = "data_BBEMI_2D_brainslice.mat"

        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"测试数据文件不存在: {file_path}")
            print("跳过MAT文件处理测试")
            return

        prim, ext = load_mat_data(file_path)
        print("成功加载MAT文件")
        print(f"主线圈数据形状: {prim.shape}")
        print(f"外部线圈数据形状: {ext.shape}")

        editer = EDITER(W=prim.shape[1])
        editer.train(prim, ext, new_kernel_size=(7, 0))

        denoise_data = editer.cancel_noise(prim, ext)

        # 检查其他测试文件
        if os.path.exists("./gksp_data.mat") and os.path.exists("./kcor_thresh.mat"):
            gksp_data = sio.loadmat("./gksp_data.mat")["gksp"]
            kpe_data = sio.loadmat("./kcor_thresh.mat")["kcor_thresh"]

            views, samples = denoise_data.shape

            img1 = reconImagesByFFT(
                denoise_data.reshape(1, 1, 1, views, 1, samples), 512
            )[0]
            img2 = reconImagesByFFT(gksp_data.reshape(1, 1, 1, views, 1, samples), 512)[
                0
            ]

            # 可选：显示图像（需要GUI环境）
            # plt.subplot(121)
            # plt.imshow(img1, cmap="grey")
            # plt.subplot(122)
            # plt.imshow(img2, cmap="grey")
            # plt.show()

            diff = denoise_data - gksp_data
            print(f"差异的L2范数: {np.linalg.norm(diff) ** 2}")
        else:
            print("参考数据文件不存在，跳过比较测试")

        print("MAT文件处理测试完成")

    except Exception as e:
        print(f"MAT文件处理测试失败: {e}")


if __name__ == "__main__":
    test_mat_processing()
