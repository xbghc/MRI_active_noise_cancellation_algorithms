"""
EDITER (External Dynamic Interference Temporal Estimation and Removal) 算法实现

该模块实现了用于MRI主动噪声消除的EDITER算法，通过分析外部线圈数据
来估计和消除主线圈中的噪声干扰。
"""

import matplotlib.pyplot as plt
import numpy as np

# 修改导入路径以适应新的包结构
from data.hyc_data_loader import HycDataLoader
from utils.mrd import reconImagesByFFT


class EDITER:
    """EDITER算法实现类"""

    def __init__(self, W, kernel_size=(0, 0), data_transpose=False):
        """初始化EDITER实例"""
        self.W = W  # 时间分组数量
        self.kernel_size = kernel_size  # 卷积核大小 (kx, ky)
        self.model = None  # 训练好的模型参数
        self.data_transpose = data_transpose  # 是否需要转置数据

    @staticmethod
    def calculate_transfer_function(interference_matrix, primary_coil):
        """
        计算传递函数h
        使用最小二乘法求解线性方程 E*h = primary_coil
        """
        h = np.linalg.lstsq(interference_matrix, primary_coil.flatten(), rcond=None)[0]
        return h

    @staticmethod
    def transpose_data(primary_matrix, external_matrices):
        """转置数据矩阵，用于处理不同的数据格式"""
        return primary_matrix.T, np.array([matrix.T for matrix in external_matrices])

    def _normalize_transfer_matrix(self, H):
        """
        归一化传递函数矩阵
        对每一列进行L2归一化，避免数值不稳定
        """
        H_normalized = np.zeros_like(H)
        for col_idx in range(H.shape[1]):
            column_norm = np.linalg.norm(H[:, col_idx])
            if column_norm > 0:
                H_normalized[:, col_idx] = H[:, col_idx] / column_norm
            else:
                H_normalized[:, col_idx] = H[:, col_idx]
        return H_normalized

    def cluster_temporal_groups(self, H, threshold=0.5):
        """
        基于相关性对时间组进行聚类
        通过计算传递函数矩阵H的相关性来确定时间组的分组范围
        """
        # 归一化传递函数矩阵，确保相关性计算的准确性
        H_normalized = self._normalize_transfer_matrix(H)

        # 计算相关性矩阵 C = H^H * H
        H_conj_T = H_normalized.conj().T
        correlation_matrix = np.dot(H_conj_T, H_normalized)

        # 根据阈值生成二值化相关性矩阵
        correlation_threshold = (np.abs(correlation_matrix) > threshold).astype(int)

        # 基于相关性矩阵确定分组范围
        groups_range = []
        left = 0
        while left < self.W:
            right = self.W - 1
            # 找到当前组的右边界
            while right > left and correlation_threshold[left, right] == 0:
                right -= 1
            groups_range.append([left, right + 1])
            left = right + 1

        return groups_range

    def _create_interference_matrix(self, external_coils):
        """
        生成干扰卷积矩阵
        将外部线圈数据转换为卷积矩阵形式，用于传递函数计算
        """
        # 计算卷积核的实际大小
        delta_kx, delta_ky = [size * 2 + 1 for size in self.kernel_size]
        kx, ky = external_coils[0].shape

        lines = []
        # 滑动窗口提取特征
        for i_x in range(kx - delta_kx + 1):
            for i_y in range(ky - delta_ky + 1):
                # 提取每个位置的卷积窗口并展平
                line = np.concatenate(
                    [
                        coil[i_x : i_x + delta_kx, i_y : i_y + delta_ky].flatten()
                        for coil in external_coils
                    ]
                )
                lines.append(line)

        return np.vstack(lines)

    def _divide_data_into_temporal_groups(
        self, primary_coil, external_coils, ranges=None
    ):
        """
        将数据按时间分组
        根据指定的范围或均匀分割将数据分成多个时间组
        """
        if ranges is None:
            # 如果没有指定范围，则均匀分割
            width = primary_coil.shape[1]
            sub_width = width // self.W
            ranges = [[i * sub_width, (i + 1) * sub_width] for i in range(self.W)]

        # 按照范围分割数据
        return [
            (primary_coil[:, start:end], external_coils[:, :, start:end])
            for start, end in ranges
        ]

    def _calculate_transfer_matrix(self, data_groups):
        """
        计算传递函数矩阵H
        对每个数据组计算传递函数，组成完整的传递函数矩阵
        """
        transfer_functions = []
        for primary_coil, external_coils in data_groups:
            # 对外部线圈数据应用填充
            padded_external = self._apply_padding(external_coils)
            # 生成干扰矩阵
            interference_matrix = self._create_interference_matrix(padded_external)
            # 计算传递函数
            h = self.calculate_transfer_function(interference_matrix, primary_coil)
            transfer_functions.append(h)

        # 将所有传递函数组合成矩阵
        return np.array(transfer_functions).reshape(len(data_groups), -1).T

    def train(self, primary_coil, external_coils, new_kernel_size=None):
        """
        训练EDITER模型
        通过分析主线圈和外部线圈的关系，学习传递函数
        """
        # 数据转置（如果需要）
        if self.data_transpose:
            primary_coil, external_coils = self.transpose_data(
                primary_coil, external_coils
            )

        # 第一步：初始分组和传递函数计算
        initial_groups = self._divide_data_into_temporal_groups(
            primary_coil, external_coils
        )
        H = self._calculate_transfer_matrix(initial_groups)

        # 第二步：基于相关性的聚类，优化分组
        correlation_ranges = self.cluster_temporal_groups(H)

        # 将聚类结果映射到实际数据范围
        width = primary_coil.shape[1] // self.W
        actual_ranges = [[i * width, j * width] for i, j in correlation_ranges]

        # 第三步：使用优化后的分组重新计算传递函数
        final_groups = self._divide_data_into_temporal_groups(
            primary_coil, external_coils, actual_ranges
        )

        # 更新卷积核尺寸（如果提供）
        if new_kernel_size is not None:
            self.kernel_size = new_kernel_size

        # 计算最终的传递函数矩阵
        final_H = self._calculate_transfer_matrix(final_groups)

        # 保存训练好的模型
        self.model = (final_H, np.array(actual_ranges))

        print(f"训练完成，共生成 {len(correlation_ranges)} 个时间组")

    def _apply_padding(self, external_coils):
        """
        对外部线圈数据应用零填充
        为卷积操作准备边界条件
        """
        pad_kx, pad_ky = self.kernel_size

        return np.array(
            [
                np.pad(
                    coil,
                    ((pad_kx, pad_kx), (pad_ky, pad_ky)),
                    mode="constant",
                    constant_values=0,
                )
                for coil in external_coils
            ]
        )

    def cancel_noise(self, primary_coil, external_coils):
        """
        使用训练好的模型进行噪声消除
        应用学习到的传递函数来预测和消除噪声
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法")

        H, ranges = self.model

        # 数据转置（如果需要）
        if self.data_transpose:
            primary_coil, external_coils = self.transpose_data(
                primary_coil, external_coils
            )

        # 按照训练时的分组方式分割数据
        data_groups = self._divide_data_into_temporal_groups(
            primary_coil, external_coils, ranges.tolist()
        )

        # 对每个数据组应用噪声消除
        cleaned_segments = []
        for idx, (primary_segment, external_segment) in enumerate(data_groups):
            # 对外部线圈数据应用填充
            padded_external = self._apply_padding(external_segment)
            # 生成干扰矩阵
            interference_matrix = self._create_interference_matrix(padded_external)
            # 使用对应的传递函数预测噪声
            predicted_noise = np.dot(interference_matrix, H[:, idx])
            # 从主线圈数据中减去预测的噪声
            cleaned_segment = primary_segment.flatten() - predicted_noise
            cleaned_segments.append(cleaned_segment.reshape(primary_segment.shape))

        # 重新组合清理后的数据段
        cleaned_primary = np.concatenate(cleaned_segments, axis=1)

        # 如果之前进行了转置，需要转置回来
        if self.data_transpose:
            cleaned_primary = cleaned_primary.T

        return cleaned_primary


def main():
    """主函数，演示EDITER算法的使用"""
    # 加载数据
    loader = HycDataLoader()
    primary_coil, external_coils = loader.load_data()

    # 创建EDITER实例
    editer = EDITER(W=8, kernel_size=(1, 1), data_transpose=True)

    # 训练模型
    print("开始训练EDITER模型...")
    editer.train(primary_coil, external_coils)

    # 应用噪声消除
    print("应用噪声消除...")
    cleaned_data = editer.cancel_noise(primary_coil, external_coils)

    # 可视化结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(reconImagesByFFT(primary_coil)), cmap="gray")
    plt.title("原始图像")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(reconImagesByFFT(cleaned_data)), cmap="gray")
    plt.title("EDITER处理后")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    difference = np.abs(reconImagesByFFT(primary_coil)) - np.abs(
        reconImagesByFFT(cleaned_data)
    )
    plt.imshow(difference, cmap="hot")
    plt.title("差异图像")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("EDITER算法演示完成")


if __name__ == "__main__":
    main()
