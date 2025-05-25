import logging
import os

import numpy as np


class HycDataLoader:
    def __init__(self, root, set_id=1):
        if not os.path.exists(root):
            logging.error("The root path does not exist!")
            return
        if not os.path.exists(os.path.join(root, "set {}".format(set_id))):
            logging.error("The set path does not exist!")
            return
        self.root = root
        self.set_id = set_id

    def load_data(self, data_type, exp_id=1):
        path = os.path.join(
            self.root, f"set {self.set_id}", f"{data_type} data", f"exp{exp_id}"
        )  # set 1，exp1注意空格
        if not os.path.exists(path):
            logging.error("The experiment path does not exist!")
            return None

        with open(os.path.join(path, f"{data_type}1.mrd"), "rb") as f:
            prim_mrd_data = self.parse_mrd(f.read())["data"][0]

        ext_mrds_data = []
        i = 2
        while True:
            data_path = os.path.join(path, f"{data_type}{i}.mrd")
            if not os.path.exists(data_path):
                break
            with open(data_path, "rb") as f:
                ext_mrds_data.append(self.parse_mrd(f.read())["data"][0])
            i += 1

        # reshape
        experiments, echoes, slices, views, views2, samples = prim_mrd_data.shape
        return [
            [
                prim_mrd_data[0, 0, 0, :, i, :],
                np.array(ext_mrds_data)[:, 0, 0, 0, :, i, :],
            ]
            for i in range(views2)
        ]

    def set_set_id(self, set_id):
        self.set_id = set_id
