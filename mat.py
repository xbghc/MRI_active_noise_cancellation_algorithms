import scipy.io as sio
import numpy as np


def load_mat_data(file_path):
    data = sio.loadmat(file_path)
    prim_key = 'datafft'
    ext_keys = ['datanoise_fft_1', 'datanoise_fft_2', 'datanoise_fft_3', 'datanoise_fft_4', 'datanoise_fft_5']

    prim_data = data[prim_key][:, :96]
    ext_data = [data[key][:, :96] for key in ext_keys]
    return np.array(prim_data), np.array(ext_data)


if __name__ == '__main__':
    from comparison import Comparison
    from EDITER import EDITER
    from mrd import DataLoader
    from yanglei import yanglei

    comparison = Comparison()

    file_path = 'data_BBEMI_2D_brainslice.mat'
    prim, ext = load_mat_data(file_path)
    comparison.add_data(prim, ext)

    comparison.add_algorithm(yanglei, 'yanglei')
    comparison.show_images()
    comparison.show_diff()

    print("Done")
