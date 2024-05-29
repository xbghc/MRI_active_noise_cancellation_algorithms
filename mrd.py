import logging
import os
import numpy as np
import matplotlib.pyplot as plt

class DataLoader:
    @staticmethod
    def parse_mrd(mrd):
        if not isinstance(mrd, bytes):
            return None
        if len(mrd) < 512:
            return None

        samples = int(0).from_bytes(mrd[0:4], byteorder='little', signed=True)
        views = int(0).from_bytes(mrd[4:8], byteorder='little', signed=True)
        views2 = int(0).from_bytes(mrd[8:12], byteorder='little', signed=True)
        slices = int(0).from_bytes(mrd[12:16], byteorder='little', signed=True)
        # 16-18 Unspecified
        datatype = int(0).from_bytes(mrd[18:20], byteorder='little', signed=True)
        # 20-152 Unspecified
        echoes = int(0).from_bytes(mrd[152:156], byteorder='little', signed=True)
        experiments = int(0).from_bytes(mrd[156:160], byteorder='little', signed=True)

        nele = experiments * echoes * slices * views * views2 * samples

        if datatype & 0xf == 0:
            dt = 'u1'
            eleSize = 1
        elif datatype & 0xf == 1:
            dt = 'i1'
            eleSize = 1
        elif datatype & 0xf == 2:
            dt = 'i2'
            eleSize = 2
        elif datatype & 0xf == 3:
            dt = 'i2'
            eleSize = 2
        elif datatype & 0xf == 4:
            dt = 'i4'
            eleSize = 4
        elif datatype & 0xf == 5:
            dt = 'f4'
            eleSize = 4
        elif datatype & 0xf == 6:
            dt = 'f8'
            eleSize = 8
        else:
            logging.error('Unknown data type in the MRD file!')
            return None
        if datatype & 0x10:
            eleSize *= 2

        #
        # XXX - The value of NO_AVERAGES in PPR cannot be used to
        #       calculate the data size.
        #       Maybe COMPLETED_AVERAGES? ref. p14 of the manual
        #
        posPPR = mrd.rfind(b'\x00')
        if posPPR == -1:
            logging.error('Corrupted MRD file!')
            return None
        posPPR += 1
        dataSize = posPPR - 512 - 120
        if dataSize < nele * eleSize:
            logging.error('Corrupted MRD file!')
            return None

        ndata = dataSize // (nele * eleSize)
        data = []

        offset = 512
        for i in range(ndata):
            x = np.frombuffer(mrd[offset:],
                              dtype=[('re', '<' + dt), ('im', '<' + dt)] if (datatype & 0x10) else ('<' + dt),
                              count=nele)
            if dt in ('f4', 'f8'):
                pass
            else:
                x = x.astype(np.float32)

            if datatype & 0x10:
                if dt in ('f8',):
                    x = x.view(np.complex128)
                else:
                    x = x.view(np.complex64)

            x = x.reshape((experiments, echoes, slices, views, views2, samples))

            offset += nele * eleSize

            data.append(x)

        if offset != posPPR - 120:
            logging.warning('Corrupted MRD file!')

        output = {}
        output['description'] = mrd[256:512].decode('cp437', errors='ignore').rstrip('\0')
        output['data'] = data
        output['sampleInfoFilePath'] = mrd[(posPPR - 120):posPPR].decode('cp437', errors='ignore').rstrip('\0')
        # output['pulseq'] = SmisPulseq(mrd[posPPR:])

        return output

    @staticmethod
    def reshape_data_to_2d(prim_coil, ext_coils):  # FIXME 重构函数
        prim_coil = np.array(prim_coil)
        ext_coils = np.array(ext_coils)

        experiments, echoes, slices, views, views2, samples = prim_coil.shape  # 手册上的views2实际发现是slices
        coil_num = ext_coils.shape[0]

        prim_coil = prim_coil.reshape((views, views2, samples))
        ext_coils = ext_coils.reshape((coil_num, views, views2, samples))
        return np.swapaxes(prim_coil, 0, 1), np.swapaxes(ext_coils, 1, 2)

    def __init__(self, root, set_id=1):
        if not os.path.exists(root):
            logging.error('The root path does not exist!')
            return
        if not os.path.exists(os.path.join(root, 'set {}'.format(set_id))):
            logging.error('The set path does not exist!')
            return
        self.root = root
        self.set_id = set_id

    def load_data(self, data_type, exp_id=1):
        path = os.path.join(self.root, f'set {self.set_id}', f'{data_type} data', f'exp{exp_id}')  # set 1，exp1注意空格
        if not os.path.exists(path):
            logging.error('The experiment path does not exist!')
            return None

        with open(os.path.join(path, f'{data_type}1.mrd'), 'rb') as f:
            all_prim_coil = self.parse_mrd(f.read())['data'][0]

        all_ext_coils = []
        i = 2
        while True:
            data_path = os.path.join(path, f'{data_type}{i}.mrd')
            if not os.path.exists(data_path):
                break
            with open(data_path, 'rb') as f:
                all_ext_coils.append(self.parse_mrd(f.read())['data'][0])
            i += 1

        all_prim_coil, all_ext_coils = self.reshape_data_to_2d(all_prim_coil, all_ext_coils)
        datas = []
        for i in range(all_prim_coil.shape[0]):
            datas.append([all_prim_coil[i, :, :], all_ext_coils[:, i, :, :]])
        return datas

    def set_set_id(self, set_id):
        self.set_id = set_id


def reconImagesByFFT(kdata, size):
        if isinstance(size, int):
            size = (size, size)

        try:
            # experiments, echoes, slices, views, views2, samples = kdata.shape
            idata = np.fft.fftshift(np.fft.ifftn(kdata, s=(size[0], kdata.shape[4], size[1]),
                                                 axes=(3, 4, 5)),
                                    axes=(3, 4, 5))
        except Exception as e:
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
