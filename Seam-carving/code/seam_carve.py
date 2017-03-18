import numpy as np
from skimage import color

w = np.array([0.299, 0.587, 0.114])

def count_energy(img_gs):
    h_grad = np.zeros_like(img_gs)
    h_grad[1:-1] = img_gs[2:] - img_gs[:-2]
    h_grad[0] = img_gs[1] - img_gs[0]
    h_grad[-1] = img_gs[-1] - img_gs[-2]

    v_grad = np.zeros_like(img_gs)
    v_grad[:, 1:-1] = img_gs[:, 2:] - img_gs[:, :-2]
    v_grad[:, 0] = img_gs[:, 1] - img_gs[:, 0]
    v_grad[:, -1] = img_gs[:, -1] - img_gs[:, -2]

    return np.sqrt(h_grad ** 2 + v_grad ** 2)

def count_seam(img_energy, MAX_ENERGY):
    seam = np.copy(img_energy)
    layers = MAX_ENERGY * np.ones((img_energy.shape[1] + 2, 3)) #, dtype=np.int)
    for i, energy in enumerate(seam[:-1]):
        layers[:-2, 0] = np.copy(energy)
        layers[1:-1, 1] = np.copy(energy)
        layers[2:, 2] = np.copy(energy)

        min_energy = np.min(layers[1:-1], axis=1)
        seam[i + 1] += min_energy

    return seam

def get_min_seam(img_seam):
    seam = np.zeros(img_seam.shape[0], dtype=np.int)
    armin = np.argmin(img_seam[-1])
    for i in range(img_seam.shape[0] - 1, -1, -1):
        if armin > 0:
            tmp = np.argmin(img_seam[i, armin - 1: armin + 2]) - 1
        else:
            tmp = np.argmin(img_seam[i, :2])
        armin += tmp
        seam[i] = armin

    return seam

def seam_carve(img, mode, mask=None):
    MAX_ENERGY = 256 * img.size
    if mode == 'vertical shrink':
        img = img.transpose(1, 0, 2)

    resized_img = np.zeros_like(img[:, :-1])
    if mask is None:
        resized_mask = None
    else:
        resized_mask = np.zeros_like(img[:, :-1])
    carve_mask = np.zeros(img.shape[:2])

    img_gray = img.dot(w)

    img_energy = count_energy(img_gray)
    img_seam = count_seam(img_energy, MAX_ENERGY)

    for i, j in enumerate(get_min_seam(img_seam)):
        carve_mask[i, j] = 1
        resized_img[i] = np.delete(img[i], j, axis=0)
        if resized_mask is not None:
            resized_mask[i] = np.delete(mask[i], j)

    if mode == 'vertical shrink':
        return (resized_img.transpose(1, 0, 2),
                resized_mask.T if resized_mask is not None else None,
                carve_mask.T)

    return (resized_img, resized_mask, carve_mask)
