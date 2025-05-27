import logging
import os

import numpy as np

from utils.mrd import parse_mrd


class HycDataLoader:
    """
    支持3维和1维数据加载
    """

    def __init__(self, root, set_id, data_type, flatten=False):
        """
        Args:
            root: 数据根目录路径
            set_id: 数据集ID
            data_type: 数据类型，只能为"noise"或"scan"
            flatten: 是否展平数据，如果为True，则返回1维数据；如果为False，则返回3维数据
        """
        # 验证data_type参数
        if data_type not in ["noise", "scan"]:
            raise ValueError(f"data_type必须为'noise'或'scan'，当前值为: {data_type}")

        if not os.path.exists(root):
            raise FileNotFoundError(f"根目录不存在: {root}")

        set_path = os.path.join(root, f"set {set_id}")
        if not os.path.exists(set_path):
            raise FileNotFoundError(f"数据集目录不存在: {set_path}")

        self.root = root
        self.set_id = set_id
        self.data_type = data_type
        self.flatten = flatten

        self.primary_data, self.external_data = self._load_all_data()

    def _load_all_data(self):
        """
        Returns:
            tuple: (primary_data, external_data)
                - primary_data: 主线圈数据，形状为 (exp, views, views2, samples)
                - external_data: 外部线圈数据，形状为 (exp, n_external_coils, views, views2, samples)
        """
        primary_data_list = []
        external_data_list = []

        for exp_id in range(1, 10):  # 最多支持9个实验
            exp_path = self._get_exp_path(exp_id)
            if not os.path.exists(exp_path):
                break

            primary_exp_data, external_exp_data = self._load_exp_data(exp_id)
            if primary_exp_data is not None and external_exp_data is not None:
                primary_data_list.append(primary_exp_data)
                external_data_list.append(external_exp_data)

        if not primary_data_list:
            raise RuntimeError("未找到有效的实验数据")

        # 转换为numpy数组，形状为 (exp, views, views2, samples)
        primary_data = np.array(primary_data_list)
        external_data = np.array(external_data_list)

        return primary_data, external_data

    def _load_exp_data(self, exp_id):
        """
        加载单个实验的数据

        Args:
            exp_id: 实验ID

        Returns:
            tuple: (primary_data, external_data) 或 (None, None)
        """
        exp_path = self._get_exp_path(exp_id)

        # 加载主线圈数据
        primary_file = os.path.join(exp_path, f"{self.data_type}1.mrd")
        if not os.path.exists(primary_file):
            logging.warning(f"主线圈数据文件不存在: {primary_file}")
            return None, None

        try:
            with open(primary_file, "rb") as f:
                primary_mrd_data = parse_mrd(f.read())["data"][0]
        except Exception as e:
            logging.error(f"读取主线圈数据失败: {e}")
            return None, None

        # 加载外部线圈数据
        external_data_list = []
        coil_idx = 2
        while True:
            external_file = os.path.join(exp_path, f"{self.data_type}{coil_idx}.mrd")
            if not os.path.exists(external_file):
                break

            try:
                with open(external_file, "rb") as f:
                    external_mrd_data = parse_mrd(f.read())["data"][0]
                    external_data_list.append(external_mrd_data)
            except Exception as e:
                logging.warning(f"读取外部线圈数据失败: {e}")
                break

            coil_idx += 1

        if not external_data_list:
            logging.warning(f"实验{exp_id}未找到外部线圈数据")
            return None, None

        # 重新整理数据格式
        # 原始数据形状: (experiments, echoes, slices, views, views2, samples)
        # 目标格式: (views, views2, samples) for primary, (n_coils, views, views2, samples) for external

        experiments, echoes, slices, views, views2, samples = primary_mrd_data.shape

        # 提取主线圈数据: (views, views2, samples)
        primary_data = primary_mrd_data[0, 0, 0, :, :, :]  # (views, views2, samples)

        # 提取外部线圈数据: (n_coils, views, views2, samples)
        external_data = np.array(
            [
                ext_data[0, 0, 0, :, :, :]  # (views, views2, samples)
                for ext_data in external_data_list
            ]
        )  # (n_coils, views, views2, samples)

        return primary_data, external_data

    def _get_exp_path(self, exp_id):
        """获取实验路径"""
        return os.path.join(
            self.root, f"set {self.set_id}", f"{self.data_type} data", f"exp{exp_id}"
        )

    def __len__(self):
        if self.flatten:
            # 展平模式：返回 n_exp * views * views2
            n_exp, views, views2 = self.primary_data.shape[:3]
            return n_exp * views * views2
        else:
            # 3维模式：返回实验数量
            return self.primary_data.shape[0]

    def __getitem__(self, index):
        """
        Returns:
            tuple: (primary_data, external_data)
                - 如果 flatten=False:
                    - primary_data: 主线圈数据，形状为 (views, views2, samples)
                    - external_data: 外部线圈数据，形状为 (n_coils, views, views2, samples)
                - 如果 flatten=True:
                    - primary_data: 主线圈数据，形状为 (samples,)
                    - external_data: 外部线圈数据，形状为 (n_coils, samples)
        """

        if self.flatten:
            # 展平模式：将线性索引转换为 (exp_idx, view_idx, views2_idx)
            n_exp, views, views2, samples = self.primary_data.shape

            # 计算三维索引
            exp_idx = index // (views * views2)
            remaining = index % (views * views2)
            view_idx = remaining // views2
            views2_idx = remaining % views2

            # 返回单行数据
            prim_row = self.primary_data[exp_idx, view_idx, views2_idx, :]  # (samples,)
            ext_row = self.external_data[
                exp_idx, :, view_idx, views2_idx, :
            ]  # (n_coils, samples)

            return prim_row, ext_row
        else:
            # 3维模式：返回整个实验的数据
            return self.primary_data[index], self.external_data[index]

    def get_data_info(self):
        """
        获取数据集信息

        Returns:
            dict: 包含数据集基本信息的字典
        """
        info = {
            "data_type": self.data_type,
            "set_id": self.set_id,
            "flatten": self.flatten,
            "n_experiments": len(self.primary_data),
            "n_external_coils": self.external_data.shape[1]
            if len(self.external_data) > 0
            else 0,
            "primary_shape": self.primary_data.shape,
            "external_shape": self.external_data.shape,
            "views": self.primary_data.shape[1] if len(self.primary_data) > 0 else 0,
            "views2": self.primary_data.shape[2] if len(self.primary_data) > 0 else 0,
            "samples": self.primary_data.shape[3] if len(self.primary_data) > 0 else 0,
        }

        if self.flatten:
            info["total_rows"] = len(self)
            info["output_primary_shape"] = f"({info['samples']},)"
            info["output_external_shape"] = (
                f"({info['n_external_coils']}, {info['samples']})"
            )
        else:
            info["output_primary_shape"] = (
                f"({info['views']}, {info['views2']}, {info['samples']})"
            )
            info["output_external_shape"] = (
                f"({info['n_external_coils']}, {info['views']}, {info['views2']}, {info['samples']})"
            )

        return info
