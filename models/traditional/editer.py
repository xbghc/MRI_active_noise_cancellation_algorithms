"""
EDITER (External Dynamic Interference Temporal Estimation and Removal) 算法实现

该模块实现了用于MRI主动噪声消除的EDITER算法，通过分析外部线圈数据
来估计和消除主线圈中的噪声干扰。
"""

import numpy as np

# 修改导入路径以适应新的包结构


class EDITER:
    """EDITER算法实现类"""

    def __init__(self, W, kernel_size=(0, 0), data_transpose=False):
        """初始化EDITER实例"""
        self.W = W  # 时间分组数量
        self.kernel_size = kernel_size  # 卷积核大小 (ky, kx)
        self.model = None  # 训练好的模型参数
        self.data_transpose = data_transpose  # 是否需要转置数据

    def calculate_transfer_function(self, ext_samples, prim_samples):
        """
        Args:
            ext_samples: 外部线圈数据
            prim_samples: 目标数据

        Returns:
            h: 传递函数，长度为 (2 * dx + 1) * (2 * dy + 1) * n_coils，其中(dy, dx) = kernel_size
        """
        convolved_ext_samples = self._convolve(ext_samples)
        h = np.linalg.lstsq(convolved_ext_samples, prim_samples.flatten(), rcond=None)[
            0
        ]
        return h

    def _normalize(self, H):
        """
        归一化传递函数矩阵，对每个h进行L2归一化

        Args:
            H: 传递函数矩阵，形状为(n_lines, n_coils * (2 * dy + 1) * (2 * dx + 1))
        """
        out = np.zeros_like(H)
        for i in range(H.shape[0]):
            normalized_h = np.linalg.norm(H[i, :])
            if normalized_h > 0:
                out[i, :] = H[i, :] / normalized_h
            else:
                out[i, :] = H[i, :]
        return out

    def _cluster(self, H, threshold=0.5):
        """
        基于相关性对时间组进行聚类
        通过计算传递函数矩阵H的相关性来确定时间组的分组范围
        """
        # 归一化传递函数矩阵，确保相关性计算的准确性
        H_normalized = self._normalize(H)

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

    def _convolve(self, ext_kdata, kernel_size=None):
        """
        将外部线圈数据转换为卷积矩阵形式，用于传递函数计算

        Args:
            ext_kdata: 外部线圈数据，kdata的形状为(coil, ky, kx)，也就是说有ky个扫描行，每个扫描行有kx个采样点
            kernel_size: 卷积核大小，包含ky和kx两个维度

        Returns:
            interference_matrix: 干扰矩阵，形状为(n_lines, delta_ky * delta_kx * n_coils)，其中n_lines是采样点的个数
        """
        if kernel_size is None:
            kernel_size = self.kernel_size

        # 计算卷积核的实际大小
        delta_ky, delta_kx = [size * 2 + 1 for size in kernel_size]

        padded_ext_kdata = self._apply_padding(ext_kdata)
        ky, kx = padded_ext_kdata[0].shape

        lines = []
        # 滑动窗口提取特征
        for i_y in range(ky - delta_ky + 1):
            for i_x in range(kx - delta_kx + 1):
                # 提取每个位置的卷积窗口并展平
                line = np.concatenate(
                    [
                        coil[i_y : i_y + delta_ky, i_x : i_x + delta_kx].flatten()
                        for coil in padded_ext_kdata
                    ]
                )
                lines.append(line)

        return np.vstack(lines)

    def _split(self, prim_kdata, ext_kdata, ranges=None):
        """
        将数据按时间分组，每组包含一个或多个扫描行，在论文中叫temporal group
        如果提供ranges，则按照ranges分割数据，否则根据设置的参数W均匀分割为W组
        """
        if ranges is None:
            width = prim_kdata.shape[1]  # TODO 这里应该按行分，而不是按列分
            sub_width = width // self.W
            ranges = [[i * sub_width, (i + 1) * sub_width] for i in range(self.W)]

        return [
            (prim_kdata[:, left:right], ext_kdata[:, :, left:right])
            for left, right in ranges
        ]

    def _calculate_transfer_funtions_of_groups(self, data_groups):
        """
        计算传递函数矩阵H
        对每个数据组计算传递函数，组成完整的传递函数矩阵
        """
        transfer_functions = []
        for prim_kdata, ext_kdata in data_groups:
            h = self.calculate_transfer_function(ext_kdata, prim_kdata)
            transfer_functions.append(h)

        return np.vstack(transfer_functions)

    def fit(self, prim_kdata, ext_kdata, new_kernel_size=None):
        """
        训练EDITER模型
        通过分析主线圈和外部线圈的关系，学习传递函数
        """

        # 第一步：初始分组和传递函数计算
        # OPTM: 这里可以每一组计算出来直接放入H矩阵中，而不是统一分组然后统一计算
        initial_groups = self._split(prim_kdata, ext_kdata)
        H = self._calculate_transfer_funtions_of_groups(initial_groups)

        # 第二步：基于相关性的聚类，优化分组
        correlation_ranges = self._cluster(H)

        # 将聚类结果映射到实际数据范围
        width = prim_kdata.shape[1] // self.W
        actual_ranges = [[i * width, j * width] for i, j in correlation_ranges]

        # 第三步：使用优化后的分组重新计算传递函数
        final_groups = self._split(prim_kdata, ext_kdata, actual_ranges)

        # 更新卷积核尺寸（如果提供）
        if new_kernel_size is not None:
            self.kernel_size = new_kernel_size

        # 计算最终的传递函数矩阵
        final_H = self._calculate_transfer_funtions_of_groups(final_groups)

        # 保存训练好的模型
        self.model = (final_H, np.array(actual_ranges))

        print(f"训练完成，共生成 {len(correlation_ranges)} 个时间组")

    def _apply_padding(self, external_coils):
        """
        对外部线圈数据应用零填充
        为卷积操作准备边界条件
        """
        pad_ky, pad_kx = self.kernel_size

        return np.array(
            [
                np.pad(
                    coil,
                    ((pad_ky, pad_ky), (pad_kx, pad_kx)),
                    mode="constant",
                    constant_values=0,
                )
                for coil in external_coils
            ]
        )

    def denoise(self, prim_kdata, ext_kdata, reTrain=False):
        """
        如果模型未训练，则先训练模型
        应用学习到的传递函数来预测和消除噪声
        """
        if self.model is None or reTrain:
            self.fit(prim_kdata, ext_kdata)

        H, ranges = self.model

        # 按照训练时的分组方式分割数据
        data_groups = self._split(prim_kdata, ext_kdata, ranges.tolist())

        # 对每个数据组应用噪声消除
        cleaned_segments = []
        for idx, (primary_segment, external_segment) in enumerate(data_groups):
            # 生成干扰矩阵
            interference_matrix = self._convolve(external_segment)
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
