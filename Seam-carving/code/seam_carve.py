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

def count_seam(img_energy):
    seam = np.copy(img_energy)
    layers = np.ones((3, img_energy.shape[1] - 2))
    for i, energy in enumerate(seam[:-1]):
        layers[0] = energy[:-2]
        layers[1] = energy[1:-1]
        layers[2] = energy[2:]

        seam[i + 1, 1:-1] += np.min(layers, axis=0)
        seam[i + 1, 0] += np.min(seam[i, :2])
        seam[i + 1, -1] += np.min(seam[i, -2:])

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

def seam_carve_one(img, mode, mask=None):
    MAX_ENERGY = 256 * img.shape[0] * img.shape[1]

    mode, action = mode.split(' ')
    resize = -1 if action == 'shrink' else 1

    if mode == 'vertical':
        img = img.transpose(1, 0, 2)
        if mask is not None:
            mask = mask.T

    resized_img = np.zeros((img.shape[0], img.shape[1] + resize, 3))
    carve_mask = np.zeros(img.shape[:2])
    if mask is None:
        resized_mask = None
    else:
        resized_mask = np.zeros((img.shape[0], img.shape[1] + resize))

    img_gray = img.dot(w)
    img_energy = count_energy(img_gray)
    if mask is not None:
        img_energy += MAX_ENERGY * mask
    img_seam = count_seam(img_energy)

    for i, j in enumerate(get_min_seam(img_seam)):
        carve_mask[i, j] = 1
        if resize == -1:  # shrink
            resized_img[i] = np.delete(img[i], j, axis=0)
        else:  # expand
            resized_img[i] = np.insert(img[i], j + 1, img[i, j], axis=0)

        if resized_mask is not None:
            if resize == -1:
                resized_mask[i] = np.delete(mask[i], j)
            else:
                resized_mask[i] = np.insert(mask[i], j + 1, mask[i, j])

    if mode == 'vertical':
        return (resized_img.transpose(1, 0, 2),
                resized_mask.T if resized_mask is not None else None,
                carve_mask.T)

    return (resized_img, resized_mask, carve_mask)

def seam_carve_object(img, mode, mask):
    """
    Needed to delete object by deleting seams one-by-one in a loop
    """
    return seam_carve_one(img, mode, mask)

def seam_carve(img, mode, mask=None):
    if mask is None:
        return seam_carve_one(img, mode)
    return seam_carve_object(img, mode, mask)
