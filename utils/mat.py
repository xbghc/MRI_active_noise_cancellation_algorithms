import numpy as np
import scipy.io as sio


def load_mat_data(file_path):
    data = sio.loadmat(file_path)
    prim_key = "datafft"
    ext_keys = [
        "datanoise_fft_1",
        "datanoise_fft_2",
        "datanoise_fft_3",
        "datanoise_fft_4",
        "datanoise_fft_5",
    ]

    prim_data = data[prim_key]
    ext_data = [data[key] for key in ext_keys]
    return np.array(prim_data), np.array(ext_data)
