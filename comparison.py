import time

import numpy as np

from mrd import reconImagesByFFT
import matplotlib.pyplot as plt
from yanglei import yanglei


class Comparison:
    def __init__(self):
        self._datas = []
        self._algorithms = {}
        self.title_setting = {
            "fontsize": 10,
            "loc": "left",
            "y": 0.8,
            "color": "white"
        }

    def add_data(self, pri_data, ext_data):
        self._datas.append([pri_data, ext_data])

    def add_algorithm(self, method, name):
        self._algorithms[name] = method

    # 展示原图
    def original_image(self, index):
        views, samples = self._datas[index][0].shape
        data = self._datas[index][0].reshape(1, 1, 1, views, 1, samples)
        return reconImagesByFFT(data, (256, 256))[0]

    def denoise_image(self, index, algorithm_name):
        denoise_kdata = self._algorithms[algorithm_name](self._datas[index][0], self._datas[index][1])
        views, samples = denoise_kdata.shape
        data = denoise_kdata.reshape(1, 1, 1, views, 1, samples)
        return reconImagesByFFT(data, (256, 256))[0]

    def diff_image(self, index, algorithm_name):
        diff = self.original_image(index) - self.denoise_image(index, algorithm_name)
        diff -= diff.min()
        return diff

    def show_images(self):
        data_size = len(self._datas)
        algorithm_size = len(self._algorithms)
        plt.figure()
        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        for col in range(data_size):
            plt.subplot(algorithm_size + 1, data_size, col + 1)
            if col == 0:
                plt.title('origin', **self.title_setting)
            plt.imshow(self.original_image(col), cmap='gray')
            plt.axis('off')

        for row, algorithm_name in enumerate(self._algorithms):
            for col in range(len(self._datas)):
                plt.subplot(algorithm_size + 1, data_size, (row + 1) * data_size + col + 1)
                if col == 0:
                    plt.title(algorithm_name, **self.title_setting)
                plt.imshow(self.denoise_image(col, algorithm_name), cmap='gray')
                plt.axis('off')

        plt.show()

    def show_diff(self):
        data_size = len(self._datas)
        algorithm_size = len(self._algorithms)

        plt.figure()
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        for row, algorithm_name in enumerate(self._algorithms):
            for col in range(len(self._datas)):
                plt.subplot(algorithm_size, data_size, row * data_size + col + 1)
                if col == 0:
                    plt.title(algorithm_name + '\'s diff', **self.title_setting)
                plt.imshow(self.diff_image(col, algorithm_name), cmap='gray')
                plt.axis('off')
        plt.show()

        plt.figure()
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        for row, algorithm_name in enumerate(self._algorithms):
            for col in range(len(self._datas)):
                plt.subplot(algorithm_size, data_size, row * data_size + col + 1)
                if col == 0:
                    plt.title(algorithm_name + '\'s diff * 10', **self.title_setting)
                plt.imshow(self.diff_image(col, algorithm_name) * 10, cmap='gray')
                plt.axis('off')
        plt.show()


if __name__ == '__main__':
    from EDITER import EDITER
    from mrd import DataLoader

    data_loader = DataLoader("datasets/HYC", set_id=4)
    primary_coil_data, external_coils_data = data_loader.load_data('noise')[0]

    editer = EDITER(W=32, window_size=(3, 1))
    editer.train(primary_coil_data, external_coils_data)

    comparison = Comparison()
    for prim, ext in data_loader.load_data('scan')[5:9]:
        comparison.add_data(prim, ext)
    comparison.add_algorithm(editer.cancel_noise, 'EDITER')
    comparison.add_algorithm(yanglei, 'yanglei')

    comparison.show_images()
    comparison.show_diff()
