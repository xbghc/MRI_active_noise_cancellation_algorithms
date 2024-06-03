import numpy as np
import matplotlib.pyplot as plt
from mrd import DataLoader, reconImagesByFFT


class EDITER:
    @staticmethod
    def calculate_h(E, prim_coil):
        h = np.linalg.pinv(E) @ prim_coil.flatten()
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

    def __init__(self, W, kernel_size=(0, 0)):
        self.W = W
        self.kernel_size = kernel_size
        self.model = None

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
            width = ext_colils.shape[1]
            if width % self.W != 0:
                print(f"kx 必须是 W 的整数倍！ kx: {width}, W: {self.W}")
                return
            sub_width = width // self.W
            ranges = [[i * sub_width, (i + 1) * sub_width] for i in range(self.W)]

        return [[prim_coil[:, i:j], ext_colils[:, :, i:j]] for i, j in ranges]

    def get_H(self, datas):
        H = []
        for prim_coil, ext_coils in datas:
            E = self.EMI_conv_matrix(ext_coils)
            prim_coil = self.trim_edges(prim_coil)
            H.append(self.calculate_h(E, prim_coil))
        H = np.array(H).reshape(len(datas), -1).T
        return H

    def get_clustered_H(self, H, prim_coil, ext_coils):
        clustered_groups_range = self.cluster(H)
        width = ext_coils.shape[1] // self.W
        clustered_groups_range = [[l * width, r * width] for l, r in clustered_groups_range]

        clustered_groups = [[prim_coil[:, i:j], ext_coils[:, :, i:j]] for i, j in clustered_groups_range]
        clustered_H = [self.calculate_h(self.EMI_conv_matrix(ext_coils), self.trim_edges(prim_coil))
                       for prim_coil, ext_coils in clustered_groups]
        return np.array(clustered_H).T, clustered_groups_range

    def train(self, prim_coil, ext_coils):
        groups = self.divide_data_into_temporal_groups(prim_coil, ext_coils)

        H = self.get_H(groups)

        H, H_range = self.get_clustered_H(H,prim_coil, ext_coils)

        self.model = H, np.array(H_range)

        print(f"get {len(H_range)} groups.")

    def trim_edges(self, prim_coil):
        ksz_x, ksz_y = self.kernel_size
        kx, ky = prim_coil.shape
        prim_coil = prim_coil[ksz_x:kx - ksz_x, ksz_y:ky - ksz_y]
        return prim_coil

    def cancel_noise(self, prim_coil, ext_coils):
        ksz_x, ksz_y = self.kernel_size
        if self.model is None:
            print("You must train the model first!")
            return
        H, H_range = self.model
        # group s_e by H_range
        data_groups = self.divide_data_into_temporal_groups(prim_coil, ext_coils, H_range)
        out = []
        for i in range(len(H_range)):
            prim_coil, ext_coils = data_groups[i]
            E = self.EMI_conv_matrix(ext_coils)
            kx, ky = prim_coil.shape
            pred_noise = np.zeros((kx, ky), dtype=np.complex64)
            flatten_pred_e = np.dot(E, H[:, i])
            pred_noise[ksz_x:kx - ksz_x,
            ksz_y:ky - ksz_y] = flatten_pred_e.reshape(
                kx - 2 * ksz_x, ky - 2 * ksz_y)
            out.append(prim_coil - pred_noise)
        return np.hstack(out)


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
