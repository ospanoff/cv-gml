import time
import glob
import numpy as np
import skimage.io as skimio
import skimage.transform as skimtr

from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout
)
from keras.optimizers import SGD
from keras.callbacks import (
    LearningRateScheduler, EarlyStopping, ModelCheckpoint
)


IMG_EDGE_SIZE = 96
IMG_SHAPE = (IMG_EDGE_SIZE, IMG_EDGE_SIZE)

GSCALE = False
INPUT_SHAPE = (*IMG_SHAPE, 1 if GSCALE else 3)  # input layer shape

Y_BIAS = IMG_EDGE_SIZE / 2
Y_NORM = IMG_EDGE_SIZE / 2
MIN_ROTATION_ANGLE = 5  # in degrees
MAX_ROTATION_ANGLE = 15  # in degrees


def rotate_img(img, y):
    alphas = list(range(-MAX_ROTATION_ANGLE, -MIN_ROTATION_ANGLE + 1)) +\
        list(range(MIN_ROTATION_ANGLE, MAX_ROTATION_ANGLE + 1))
    # alpha = 2 * MAX_ROTATE_ANGLE * (np.random.rand() - 0.5)
    alpha = np.random.choice(alphas)
    alpha_rad = np.radians(alpha)
    rot_mat = np.array([[np.cos(alpha_rad), -np.sin(alpha_rad)],
                        [np.sin(alpha_rad), np.cos(alpha_rad)]])
    bias = img.shape[0] / 2
    return (
        skimtr.rotate(img, alpha),
        (y - bias).reshape(-1, 2).dot(rot_mat).ravel() + bias
    )


def cut_img(img, y):
    h = img.shape[0]
    lt = int(np.ceil(min(np.random.randint(0.05 * h, 0.15 * h), y.min())))
    rb = int(np.ceil(max(np.random.randint(0.85 * h, 0.95 * h), y.max())))
    return img[lt: rb, lt: rb], y - lt


def flip_img(img, y):
    y_ = y.copy()
    y_[::2] = img.shape[1] - y_[::2] - 1
    return (
        img[:, ::-1],
        y_.reshape(-1, 2)[
            [3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 10, 13, 12, 11]
        ].ravel()
    )


def load_data(img_dir, gt, input_shape, output_size=28, test=False):
    print("STARTED LOADING DATA")
    _start = time.time()

    N = len(gt)
    rotations_num = 2
    cut_num = 1

    start_flipped = N
    start_rotation = start_flipped + N
    start_cut = start_rotation + rotations_num * N

    if not test:
        N = 2 * N + rotations_num * N + cut_num * N  # one N for flipped imgs
    X = np.empty((N, *input_shape))
    y = np.empty((N, output_size)) if not test else None
    scales = []

    for i, (fn, y_raw) in enumerate(gt.items()):
        img = skimio.imread(img_dir + '/' + fn, as_grey=GSCALE)
        scale_y = 1.0 * img.shape[0] / input_shape[0]
        scale_x = 1.0 * img.shape[1] / input_shape[1]
        scales += [(scale_x, scale_y, fn)]

        X[i] = skimtr.resize(img, input_shape, mode='reflect')

        if not test:
            # Original image
            y[i][::2] = y_raw[::2] / scale_x
            y[i][1::2] = y_raw[1::2] / scale_y

            # Flipped image
            X[start_flipped + i], y[start_flipped + i] = flip_img(X[i], y[i])

            # Rotated images
            for r in range(rotations_num):
                indx = start_rotation + rotations_num * i + r
                X[indx], y[indx] = rotate_img(X[i], y[i])

            # Cutted images
            for c in range(cut_num):
                indx = start_cut + cut_num * i + c
                if y_raw.min() < 0:
                    X[indx], y[indx] = X[i], y[i]
                else:
                    img_c, y_c = cut_img(img, y_raw)
                    scale = 1.0 * img_c.shape[0] / input_shape[0]
                    X[indx] = skimtr.resize(img_c, input_shape, mode='reflect')
                    y[indx] = y_c / scale

    if not test:
        y = (y - Y_BIAS) / Y_NORM

    # mean = np.mean(X, axis=0)
    # std = (np.mean(X ** 2, axis=0) - mean ** 2) ** 0.5
    # X = (X - mean) / std

    print("FINISHED LOADING DATA:", time.time() - _start)

    return X, y, scales


def build_model(input_shape, output_size=28):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(output_size))

    return model


def train_detector(train_gt, train_img_dir, fast_train=False):
    input_shape = INPUT_SHAPE  # input layer shape
    output_size = len(list(train_gt.values())[0])
    if fast_train:
        keys = list(train_gt.keys())[:10]
        train_gt = {key: train_gt[key] for key in keys}

    X, y, _ = load_data(train_img_dir, train_gt, input_shape, output_size)

    # Model config.
    epochs = 1 if fast_train else 1000
    learning_rates = np.linspace(0.03, 0.001, epochs)
    patience = 100  # stop if err has not been updated patience time

    change_lr = LearningRateScheduler(lambda epoch: learning_rates[epoch])
    early_stop = EarlyStopping(patience=patience)

    # SGD config.
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

    # Model setup
    model = build_model(input_shape, output_size)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    checkpoint_callback = ModelCheckpoint(filepath='model.hdf5',
                                          monitor='val_loss',
                                          save_best_only=True,
                                          mode='auto')

    # Model training
    start_time = time.time()
    print('start_time: {}'.format(time.strftime('%H:%M:%S')))
    model.fit(
        X, y,
        epochs=epochs,
        batch_size=1000 if not fast_train else 32,
        validation_split=0.2,
        callbacks=[early_stop, change_lr, checkpoint_callback]
    )
    print('end_time: {}, duration(min): {}'.format(time.strftime('%H:%M:%S'),
          (time.time()-start_time) / 60.))

    return model


def detect(model, test_img_dir):
    gt = {}
    for fname in glob.glob1(test_img_dir, '*.jpg'):
        gt[fname] = None

    X, _, scales = load_data(test_img_dir, gt, INPUT_SHAPE, test=True)
    y_pred = model.predict(X) * Y_NORM + Y_BIAS

    y_scaled = {}
    for i in range(len(scales)):
        scale_x, scale_y, fn = scales[i]
        y = y_pred[i]
        y[::2] = y[::2] * scale_x
        y[1::2] = y[1::2] * scale_y
        y_scaled[fn] = y

    return y_scaled
