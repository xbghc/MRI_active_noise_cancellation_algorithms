import numpy as np
import scipy.io as sio


def load_mat_data(file_path):
    data = sio.loadmat(file_path)
    prim_key = "datafft"
    ext_keys = [
        "datanoise_fft_1",
        "datanoise_fft_2",
        "datanoise_fft_3",
        "datanoise_fft_4",
        "datanoise_fft_5",
    ]

    prim_data = data[prim_key]
    ext_data = [data[key] for key in ext_keys]
    return np.array(prim_data), np.array(ext_data)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from traditional.editer import EDITER
    from utils.mrd import reconImagesByFFT

    file_path = "data_BBEMI_2D_brainslice.mat"
    prim, ext = load_mat_data(file_path)

    editer = EDITER(W=prim.shape[1])
    editer.train(prim, ext, new_kernel_size=(7, 0))

    denoise_data = editer.cancel_noise(prim, ext)
    gksp_data = sio.loadmat("./gksp_data.mat")["gksp"]
    kpe_data = sio.loadmat("./kcor_thresh.mat")["kcor_thresh"]

    views, samples = denoise_data.shape

    img1 = reconImagesByFFT(denoise_data.reshape(1, 1, 1, views, 1, samples), 512)[0]
    img2 = reconImagesByFFT(gksp_data.reshape(1, 1, 1, views, 1, samples), 512)[0]

    plt.subplot(121)
    plt.imshow(img1, cmap="grey")
    plt.subplot(122)
    plt.imshow(img2, cmap="grey")
    plt.show()

    diff = denoise_data - gksp_data
    # print(diff)
    print(np.linalg.norm(diff) ** 2)

    # comparison = Comparison()
    # comparison.add_data(prim, ext)
    # comparison.add_algorithm(yanglei, 'yanglei')
    # comparison.add_algorithm(editer.cancel_noise, 'EDITER')
    # comparison.show_images()
    # comparison.show_diff()
    print("Done")
