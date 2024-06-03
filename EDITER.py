import numpy as np
import matplotlib.pyplot as plt
from mrd import DataLoader, reconImagesByFFT


class EDITER:
    @staticmethod
    def calculate_h(E, prim_coil):
        h = np.linalg.lstsq(E, prim_coil.flatten(), rcond=None)[0]

        return h

    @staticmethod
    def transpose_data(prim_matrix, ext_matriex):
        return prim_matrix.T, np.array(
            [matrix.T for matrix in ext_matriex]
        )

    def cluster(self, H, threshold=0.5):
        H_normalized = np.zeros_like(H)
        for clin in range(H.shape[1]):
            H_normalized[:, clin] = H[:, clin] / np.linalg.norm(H[:, clin])
        H = H_normalized

        H_conj_T = H.conj().T
        C = np.dot(H_conj_T, H)
        C_threshold = (np.abs(C) > threshold).astype(int)

        groups_range = []
        l = 0
        while l < self.W:
            r = self.W - 1
            while r > l and C_threshold[l, r] == 0:
                r -= 1
            groups_range.append([l, r + 1])
            l = r + 1
        return groups_range

    def __init__(self, W, kernel_size=(0, 0), data_transpose=False):
        self.W = W
        self.kernel_size = kernel_size
        self.model = None
        self.data_transpose = data_transpose

    # 生成EMI_conv_matrix
    def EMI_conv_matrix(self, ext_coils):
        delta_kx, delta_ky = [_ * 2 + 1 for _ in self.kernel_size]
        lines = []
        kx, ky = ext_coils[0].shape
        if kx < delta_kx or ky < delta_ky:
            print(f'error: kx({kx}) < delta_kx({delta_kx}) or ky({ky}) < delta_ky({delta_ky})')
            return

        for i_x in range(kx - delta_kx + 1):
            for i_y in range(ky - delta_ky + 1):
                line = np.concatenate(tuple(
                    d[i_x:i_x + delta_kx, i_y:i_y + delta_ky].flatten() for d in ext_coils
                ))
                lines.append(line)
        E = np.vstack(tuple(lines))
        return E

    def divide_data_into_temporal_groups(self, prim_coil, ext_colils, ranges=None):
        if ranges is None:
            width = prim_coil.shape[1]
            if width % self.W != 0:
                print(f"ky 必须是 W 的整数倍！ ky: {width}, W: {self.W}")
                return
            sub_width = width // self.W
            ranges = [[i * sub_width, (i + 1) * sub_width] for i in range(self.W)]

        return [[prim_coil[:, i:j], ext_colils[:, :, i:j]] for i, j in ranges]

    def get_H(self, datas):
        H = []
        for prim_coil, ext_coils in datas:
            E = self.EMI_conv_matrix(self.padding(ext_coils))
            H.append(self.calculate_h(E, prim_coil))
        H = np.array(H).reshape(len(datas), -1).T
        return H

    def train(self, prim_coil, ext_coils, new_kernel_size=None):
        if self.data_transpose:
            prim_coil, ext_coils = self.transpose_data(prim_coil, ext_coils)

        groups = self.divide_data_into_temporal_groups(prim_coil, ext_coils)
        H = self.get_H(groups)

        correlation_range = self.cluster(H)
        width = prim_coil.shape[1] // self.W
        correlation_range = [[i*width, j*width] for i, j in correlation_range]

        groups = self.divide_data_into_temporal_groups(prim_coil, ext_coils, correlation_range)

        if new_kernel_size:
            self.kernel_size = new_kernel_size
        H = self.get_H(groups)

        self.model = H, np.array(correlation_range)

        print(f"get {len(correlation_range)} groups.")

    def padding(self, ext_coils):
        ksz_x, ksz_y = self.kernel_size

        return np.array([np.pad(coil, ((ksz_x, ksz_x), (ksz_y, ksz_y)), mode='constant', constant_values=0) for coil in ext_coils])

    def cancel_noise(self, prim_coil, ext_coils):
        if self.data_transpose:
            prim_coil, ext_coils = self.transpose_data(prim_coil, ext_coils)

        if self.model is None:
            print("You must train the model first!")
            return

        H, H_range = self.model
        # group s_e by H_range
        data_groups = self.divide_data_into_temporal_groups(prim_coil, ext_coils, H_range)
        out = []
        for i in range(len(H_range)):
            prim_coil, ext_coils = data_groups[i]
            ext_coils = self.padding(ext_coils)
            E = self.EMI_conv_matrix(ext_coils)
            kx, ky = prim_coil.shape
            flatten_pred_e = np.dot(E, H[:, i])
            pred_noise = flatten_pred_e.reshape(kx, ky)
            out.append(prim_coil - pred_noise)
        out = np.hstack(out)

        if self.data_transpose:
            out = out.T
        return out



if __name__ == '__main__':
    # FIXME 并未做T2成像所得.mrd文件的数据处理

    data_loader = DataLoader("datasets/HYC", set_id=4)
    primary_coil_data, external_coils_data = data_loader.load_data('noise')[0]

    editer = EDITER(W=32)
    editer.train(primary_coil_data, external_coils_data)

    image_index = 5
    primary_coil_data, external_coils_data = data_loader.load_data('scan')[image_index]

    s = editer.cancel_noise(primary_coil_data, external_coils_data)

    x, y = primary_coil_data.shape
    noise_img = reconImagesByFFT(primary_coil_data.reshape(1, 1, 1, x, 1, y), 256)[0]

    no_noise_img = reconImagesByFFT(s.reshape(1, 1, 1, x, 1, y), 256)[0]
    plt.subplot(1, 2, 1)
    plt.imshow(noise_img, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(no_noise_img, cmap='gray')
    plt.show()

    print("Done!")
