import logging

import numpy as np


def parse_mrd(mrd):
    if not isinstance(mrd, bytes):
        return None
    if len(mrd) < 512:
        return None

    samples = int(0).from_bytes(mrd[0:4], byteorder="little", signed=True)
    views = int(0).from_bytes(mrd[4:8], byteorder="little", signed=True)
    views2 = int(0).from_bytes(mrd[8:12], byteorder="little", signed=True)
    slices = int(0).from_bytes(mrd[12:16], byteorder="little", signed=True)
    # 16-18 Unspecified
    datatype = int(0).from_bytes(mrd[18:20], byteorder="little", signed=True)
    # 20-152 Unspecified
    echoes = int(0).from_bytes(mrd[152:156], byteorder="little", signed=True)
    experiments = int(0).from_bytes(mrd[156:160], byteorder="little", signed=True)

    nele = experiments * echoes * slices * views * views2 * samples

    if datatype & 0xF == 0:
        dt = "u1"
        eleSize = 1
    elif datatype & 0xF == 1:
        dt = "i1"
        eleSize = 1
    elif datatype & 0xF == 2:
        dt = "i2"
        eleSize = 2
    elif datatype & 0xF == 3:
        dt = "i2"
        eleSize = 2
    elif datatype & 0xF == 4:
        dt = "i4"
        eleSize = 4
    elif datatype & 0xF == 5:
        dt = "f4"
        eleSize = 4
    elif datatype & 0xF == 6:
        dt = "f8"
        eleSize = 8
    else:
        logging.error("Unknown data type in the MRD file!")
        return None
    if datatype & 0x10:
        eleSize *= 2

    #
    # XXX - The value of NO_AVERAGES in PPR cannot be used to
    #       calculate the data size.
    #       Maybe COMPLETED_AVERAGES? ref. p14 of the manual
    #
    posPPR = mrd.rfind(b"\x00")
    if posPPR == -1:
        logging.error("Corrupted MRD file!")
        return None
    posPPR += 1
    dataSize = posPPR - 512 - 120
    if dataSize < nele * eleSize:
        logging.error("Corrupted MRD file!")
        return None

    ndata = dataSize // (nele * eleSize)
    data = []

    offset = 512
    for i in range(ndata):
        x = np.frombuffer(
            mrd[offset:],
            dtype=[("re", "<" + dt), ("im", "<" + dt)]
            if (datatype & 0x10)
            else ("<" + dt),
            count=nele,
        )
        if dt in ("f4", "f8"):
            pass
        else:
            x = x.astype(np.float32)

        if datatype & 0x10:
            if dt in ("f8",):
                x = x.view(np.complex128)
            else:
                x = x.view(np.complex64)

        x = x.reshape((experiments, echoes, slices, views, views2, samples))

        offset += nele * eleSize

        data.append(x)

    if offset != posPPR - 120:
        logging.warning("Corrupted MRD file!")

    output = {}
    output["description"] = mrd[256:512].decode("cp437", errors="ignore").rstrip("\0")
    output["data"] = data
    output["sampleInfoFilePath"] = (
        mrd[(posPPR - 120) : posPPR].decode("cp437", errors="ignore").rstrip("\0")
    )
    # output['pulseq'] = SmisPulseq(mrd[posPPR:])

    return output


def reconImagesByFFT(kdata, size):
    if isinstance(size, int):
        size = (size, size)

    try:
        # experiments, echoes, slices, views, views2, samples = kdata.shape
        idata = np.fft.fftshift(
            np.fft.ifftn(kdata, s=(size[0], kdata.shape[4], size[1]), axes=(3, 4, 5)),
            axes=(3, 4, 5),
        )
    except Exception:
        return []

    idata = np.abs(idata)
    idata /= idata.max()

    images = []
    for e in range(idata.shape[0]):
        for k in range(idata.shape[1]):
            for s in range(idata.shape[2]):
                for v2 in range(idata.shape[4]):
                    images.append(idata[e, k, s, :, v2, :])

    return images


# class DataVisualizer:
#     @staticmethod
#     def reconImagesByFFT(kdata, size):
#         if isinstance(size, int):
#             size = (size, size)
#
#         try:
#             # experiments, echoes, slices, views, views2, samples = kdata.shape
#             idata = np.fft.fftshift(np.fft.ifftn(kdata, s=(size[0], kdata.shape[4], size[1]),
#                                                  axes=(3, 4, 5)),
#                                     axes=(3, 4, 5))
#         except Exception as e:
#             return []
#
#         idata = np.abs(idata)
#         idata /= idata.max()
#
#         images = []
#         for e in range(idata.shape[0]):
#             for k in range(idata.shape[1]):
#                 for s in range(idata.shape[2]):
#                     for v2 in range(idata.shape[4]):
#                         images.append(idata[e, k, s, :, v2, :])
#
#         return images
#
#     @staticmethod
#     def visualize_kdata(self, kdata, size):
#         images = self.reconImagesByFFT(kdata, size)
#         # use plt to show
#         for i in range(len(images)):
#             plt.subplot(1, len(images), i+1)
#             plt.imshow(images[i], cmap='gray')
#         plt.show()
#
#     @staticmethod
#     def visualize_2d_data(self, data, size):
#         data = data.reshape((1, 1, 1, data.shape[0], data.shape[1], data.shape[2]))
#         kdata = np.swapaxes(data, 3, 4)
#         self.visualize_kdata(kdata, size)
#
#     def __init__(self):
#         pass
