import numpy as np


class Yanglei:
    """
    yanglei算法本是用于实时降噪，不需要根据空采数据进行训练。这里提供了使用空采数据训练的接口
    - 有一定的性能损失，fit和denoise会分别将数据拆分为多个bin
    - 原方法傅里叶变换后没有调用fftshift，这里会调用
    """

    def __init__(self, nbin=8, v=0):
        """
        Args:
            nbin: 频率分组的数量
            v: 默认为0，本意是取最边缘的数据，用于近似空采数据
        """
        self.nbin = nbin
        self.v = v

        self.c = None  # 系数矩阵

    def fit(self, prim_coil, ext_coils):
        """
        支持各种维度的数据，计算方法是一样的
        """
        n_nos = len(ext_coils)

        idata_obj = np.fft.fftshift(np.fft.fft(prim_coil))
        idata_nos = np.fft.fftshift(np.fft.fft(ext_coils))

        self.c = []
        obj_bins = np.split(idata_obj, self.nbin, axis=-1)
        nos_bins = np.split(idata_nos, self.nbin, axis=-1)

        for i in range(self.nbin):
            X = nos_bins[i].swapaxes(0, -1).reshape(-1, n_nos)
            y = obj_bins[i].flatten()
            self.c.append(np.linalg.lstsq(X, y, rcond=None))

    def predict(self, prim_coil, ext_coils, reTrain):
        if reTrain or self.c is None:
            self.fit(prim_coil, ext_coils)

        idata_obj = np.fft.fftshift(np.fft.fft(prim_coil))
        idata_nos = np.fft.fftshift(np.fft.fft(ext_coils))

        nos_bins = np.split(idata_nos, self.nbin, axis=-1)
        obj_bins = np.split(idata_obj, self.nbin, axis=-1)
        # 预测结果
        result = np.zeros_like(idata_obj)
        for i in range(self.nbin):
            X = nos_bins[i].swapaxes(0, -1).reshape(-1, len(ext_coils))
            predicted_y = X @ self.c[i][0]
            result_bin = predicted_y.reshape(obj_bins[i].shape)
            result_bin = result_bin.reshape(nos_bins[i].shape[1:])
            result_bin_start = i * (result.shape[-1] // self.nbin)
            result_bin_end = (i + 1) * (result.shape[-1] // self.nbin)
            result[..., result_bin_start:result_bin_end] = result_bin

        # 转回时域
        return np.fft.ifft(np.fft.ifftshift(result))

    def denoise(self, prim_coil, ext_coils, reTrain):
        return prim_coil - self.predict(prim_coil, ext_coils, reTrain)
