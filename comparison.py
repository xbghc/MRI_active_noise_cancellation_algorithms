import time

from mrd import reconImagesByFFT
import matplotlib.pyplot as plt
from yanglei import yanglei


class Comparison:
    def __init__(self):
        self._datas = []
        self._algorithms = {}

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

    def show(self):
        data_size = len(self._datas)
        algorithm_size = len(self._algorithms)
        title_size = 100
        # plt.subplots_adjust(top=10)

        for j in range(data_size):
            plt.figure(figsize=(10 * (algorithm_size + 1), 10))
            plt.subplot(1, algorithm_size+1, 1)
            plt.title('origin', fontsize=title_size)

            plt.imshow(self.original_image(j), cmap='gray')

            i = 1
            for algorithm_name in self._algorithms:
                plt.subplot(1, algorithm_size+1, i+1)
                plt.title(algorithm_name, fontsize=title_size)
                plt.imshow(self.denoise_image(j, algorithm_name), cmap='gray')
                i += 1

            plt.tight_layout()
            plt.show()
            time.sleep(1)


if __name__ == '__main__':
    from EDITER import EDITER
    from mrd import DataLoader

    data_loader = DataLoader("datasets/HYC", set_id=4)
    primary_coil_data, external_coils_data = data_loader.load_data('noise')[0]

    editer = EDITER(W=32, window_size=(3, 1))
    editer.train(primary_coil_data, external_coils_data)

    comparison = Comparison()
    for prim, ext in data_loader.load_data('scan'):
        comparison.add_data(prim, ext)
    comparison.add_algorithm(editer.cancel_noise, 'EDITER')
    comparison.add_algorithm(yanglei, 'yanglei')

    comparison.show()
