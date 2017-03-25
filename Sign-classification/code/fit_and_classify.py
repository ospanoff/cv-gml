import numpy as np
from sklearn.svm import SVC
from skimage.transform import resize

def count_energy(img_gs):
    h_grad = np.zeros_like(img_gs)
    h_grad[1:-1] = img_gs[2:] - img_gs[:-2]
    h_grad[0] = img_gs[1] - img_gs[0]
    h_grad[-1] = img_gs[-1] - img_gs[-2]

    v_grad = np.zeros_like(img_gs)
    v_grad[:, 1:-1] = img_gs[:, 2:] - img_gs[:, :-2]
    v_grad[:, 0] = img_gs[:, 1] - img_gs[:, 0]
    v_grad[:, -1] = img_gs[:, -1] - img_gs[:, -2]

    return np.sqrt(h_grad ** 2 + v_grad ** 2), np.arctan2(h_grad   , v_grad)

def hog(image_gs, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(3, 3), ignore_extra=False):
    img_energy, img_dir = count_energy(image_gs)

    if ignore_extra:
        cells_h, cells_w = np.round(np.array(image_gs.shape) / pixels_per_cell).astype(np.int)
    else:
        cells_h, cells_w = np.ceil(np.array(image_gs.shape) / pixels_per_cell).astype(np.int)
    hog_feat = np.zeros((cells_h, cells_w, orientations))

    ppc_h, ppc_w = pixels_per_cell
    for i in range(image_gs.shape[0]):
        for j in range(image_gs.shape[1]):
            segment = int((orientations * (img_dir[i, j] + np.pi)) / (2 * np.pi))
            try:
                hog_feat[i // ppc_h, j // ppc_w, segment] += img_energy[i, j]
            except:
                pass

    cpb_h, cpb_w = cells_per_block
    hog_final = []
    for i in range(hog_feat.shape[0] - cpb_h + 1):
        for j in range(hog_feat.shape[1] - cpb_w + 1):
            v = hog_feat[i: i + cpb_h, j: j + cpb_w].ravel()
            hog_final += (v / np.sqrt((v ** 2).sum() + 1e-30)).tolist()

    return np.array(hog_final)

def extract_hog(img, shape=(45, 45)):
    img_gs = img.dot([0.299, 0.587, 0.114])
    return hog(resize(img_gs, shape))

def fit_and_classify(train_featues, train_labels, test_features):
    clf = OneVsRestClassifier(SVC(C=300, gamma=0.01), n_jobs=-1)
    clf.fit(train_featues, train_labels)
    return clf.predict(test_features)
