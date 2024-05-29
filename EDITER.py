import numpy as np
import matplotlib.pyplot as plt
from mrd import DataLoader, reconImagesByFFT


class EDITER:
    @staticmethod
    def calculate_h(E, e):
        h = np.linalg.pinv(E) @ e.flatten()
        return h

    def cluster(self, H, threshold=0.5):
        # normalize
        H = H / np.linalg.norm(H, axis=0)

        H_conj_T = H.conj().T
        C = np.dot(H_conj_T, H)
        C_threshold = (C >= threshold).astype(int)

        groups_range = []
        r = 0
        while r < self.W:
            l = r
            while r < self.W and C_threshold[l, r] == 1:
                r += 1
            groups_range.append([l, r])
        return groups_range

    def __init__(self, W, window_size=(3, 1)):
        self.W = W
        self.window_size = window_size
        self.model = None

    # 生成EMI_conv_matrix
    def EMI_conv_matrix(self, detectors):
        delta_kx, delta_ky = self.window_size
        lines = []
        kx, ky = detectors[0].shape
        if kx < delta_kx or ky < delta_ky:
            print(f"detectors的大小必须大于等于window_size! kx: {kx}, ky: {ky}, delta_kx: {delta_kx}, delta_ky: {delta_ky}")
            return

        for i_x in range(kx - delta_kx + 1):
            for i_y in range(ky - delta_ky + 1):
                line = np.concatenate(tuple(
                    d[i_x:i_x + delta_kx, i_y:i_y + delta_ky].flatten() for d in detectors
                ))
                lines.append(line)
        E = np.vstack(tuple(lines))
        return E

    def divide_data_into_temporal_groups(self, detectors, e, ranges=None):
        if ranges is None:
            width = detectors.shape[1]
            if width % self.W != 0:
                print(f"kx 必须是 W 的整数倍！ kx: {width}, W: {self.W}")
                return
            sub_width = width // self.W
            ranges = [[i*sub_width, (i+1)*sub_width] for i in range(self.W)]

        return [[detectors[:, :, i:j], e[:, i:j]] for i, j in ranges]

    def cut_e(self, e):
        delta_kx, delta_ky = self.window_size
        kx, ky = e.shape
        e = e[(delta_kx-1)//2:kx-(delta_kx-1)//2, (delta_ky-1)//2:ky-(delta_ky-1)//2]
        return e

    def get_H(self, datas):
        H = []
        for detectors_in_group, e_in_group in datas:
            E = self.EMI_conv_matrix(detectors_in_group)
            e = self.trim_edges(e_in_group)
            H.append(self.calculate_h(E, e))
        H = np.array(H).reshape(len(datas), -1).T
        return H

    def get_clustered_H(self, H, detectors, e):
        clustered_groups_range = self.cluster(H)
        width = detectors.shape[1] // self.W
        clustered_groups_range = [[l * width, r * width] for l, r in clustered_groups_range]

        clustered_groups = [[detectors[:, :, i:j], e[:, i:j]] for i, j in clustered_groups_range]
        clustered_H = [self.calculate_h(self.EMI_conv_matrix(detectors_in_group), self.trim_edges(e_in_group))
                       for detectors_in_group, e_in_group in clustered_groups]
        return np.array(clustered_H).T, clustered_groups_range

    def train(self, detectors, e):
        groups = self.divide_data_into_temporal_groups(detectors, e)

        H = self.get_H(groups)

        H, H_range = self.get_clustered_H(H, detectors, e)

        self.model = H, np.array(H_range)

    def trim_edges(self, e):
        delta_kx, delta_ky = self.window_size
        kx, ky = e.shape
        e = e[(delta_kx-1)//2:kx-(delta_kx-1)//2, (delta_ky-1)//2:ky-(delta_ky-1)//2]
        return e


    def cancel_noise(self, s_e, detectors):
        if self.model is None:
            print("You must train the model first!")
            return
        H, H_range = self.model
        # group s_e by H_range
        data_groups = self.divide_data_into_temporal_groups(detectors, s_e, H_range)
        out = []
        for i in range(len(H_range)):
            detectors_in_group, s_e_in_group = data_groups[i]
            E = self.EMI_conv_matrix(detectors_in_group)
            kx, ky = s_e_in_group.shape
            pred_e = np.zeros((kx, ky), dtype=np.complex64)
            flatten_pred_e = np.dot(E, H[:, i])
            pred_e[self.window_size[0]//2:kx-self.window_size[0]//2, self.window_size[1]//2:ky-self.window_size[1]//2] = flatten_pred_e.reshape(kx-self.window_size[0]+1, ky-self.window_size[1]+1)
            out.append(s_e_in_group - pred_e)
        return np.hstack(out)


if __name__ == '__main__':
    # FIXME 并未做T2成像所得.mrd文件的数据处理

    data_loader = DataLoader("datasets/HYC", set_id=4)
    primary_coil_data, extermal_coils_data = data_loader.load_train_data()

    editer = EDITER(W=32, window_size=(3, 1))
    editer.train(extermal_coils_data[:, 0, :, :], primary_coil_data[0])

    primary_coil_data, extermal_coils_data = data_loader.load_test_data()
    image_index = 5
    s_and_e = primary_coil_data[image_index]
    s = editer.cancel_noise(s_and_e, extermal_coils_data[:, image_index,:,:])

    x, y = s_and_e.shape
    noise_img = reconImagesByFFT(s_and_e.reshape(1, 1, 1, x, 1, y), 256)[0]

    no_noise_img = reconImagesByFFT(s.reshape(1, 1, 1, x, 1, y), 256)[0]
    plt.subplot(1, 2, 1)
    plt.imshow(noise_img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(no_noise_img, cmap='gray')
    plt.show()
 
    print("Done!")