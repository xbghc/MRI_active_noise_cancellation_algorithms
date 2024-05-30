import numpy as np
import logging

def _yanglei(kdatas, nbin=8, v=0):
    kdata_obj = kdatas[0]
    kdata_nos = [_ for _ in kdatas[1:] if _ is not None]

    if len(kdata_nos) <= 0:
        return kdata_obj

    if (not isinstance(nbin, int)) or nbin <= 0:
        logging.warning('The nbin(=%d) must be an integer greater than 0.' % (nbin,))
        return kdata_obj

    npb = kdata_nos[0].shape[5]//nbin

    if (nbin*npb) != kdata_nos[0].shape[5]:
        logging.warning('The nbin(=%d) must be divisible by number of samples(=%d).' % (nbin, kdata_nos[0].shape[5]))
    else:
        idata_obj = np.fft.fft(kdata_obj)
        idata_nos = np.array([np.fft.fft(kdata_nos[i]) for i in range(len(kdata_nos))])

        X = idata_nos[:, 0, 0, 0, v, 0, :]
        y = idata_obj[   0, 0, 0, v, 0, :]
        c = np.zeros((idata_nos.shape[0], nbin), dtype=y.dtype)

        for i in range(c.shape[1]):
            rng = range(i*npb, (i+1)*npb)
            _ = np.linalg.lstsq(X[:, rng].T, y[rng], rcond=None)
            c[:, i] = _[0]

        idata_obj -= np.reshape(np.sum(c[:, None, None, None, None, None, :, None] *
                                       idata_nos.reshape((idata_nos.shape[0], idata_nos.shape[1], idata_nos.shape[2],
                                                          idata_nos.shape[3], idata_nos.shape[4], idata_nos.shape[5],
                                                          c.shape[1], -1)),
                                       axis=0),
                                idata_obj.shape)

        kdata_obj = np.fft.ifft(idata_obj)

    return kdata_obj

def yanglei(prim_coil, ext_coils, nbin=8, v=0):
    views, samples = prim_coil.shape
    prim_coil = prim_coil.reshape(1, 1, 1, views, 1, samples)
    ext_coils = ext_coils.reshape(-1, 1, 1, 1, views, 1, samples)
    kdatas = [prim_coil, *ext_coils]
    return _yanglei(kdatas, nbin, v).reshape(views, samples)

