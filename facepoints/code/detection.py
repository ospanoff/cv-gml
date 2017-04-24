import time
import glob
import numpy as np
import skimage.io as skio
import skimage.transform as sktr
import sklearn.utils as skut

from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Activation, Dropout
)
from keras.optimizers import SGD
from keras.callbacks import (
    LearningRateScheduler, EarlyStopping, ModelCheckpoint
)


Y_MEAN = 42.
Y_STD = 19.
IMG_SHAPE = (100, 100)


def load_data(img_dir, gt, input_shape, output_size=28, test=False):
    X = np.empty((len(gt), *input_shape))
    y = np.empty((len(gt), output_size)) if not test else None
    scales = []

    for i, (fn, y_) in enumerate(gt.items()):
        img = skio.imread(img_dir + '/' + fn, as_grey=True)
        scale_y = img.shape[0] / input_shape[0]
        scale_x = img.shape[1] / input_shape[1]
        scales += [(scale_x, scale_y, fn)]

        X[i] = sktr.resize(img, input_shape[:2]).reshape(input_shape)

        if not test:
            y_[::2] = y_[::2] / scale_x
            y_[1::2] = y_[1::2] / scale_y
            y[i] = y_

    mean = np.mean(X, axis=0)
    std = (np.mean(X**2, axis=0) - mean**2) ** 0.5

    X = (X - mean) / std

    if not test:
        X_fl = X[:, :, ::-1].copy()
        y_fl = y.copy()
        y_fl[:, ::2] = input_shape[1] - y_fl[:, ::2] - 1

        X, y = skut.shuffle(np.vstack((X, X_fl)), np.vstack((y, y_fl)),
                            random_state=42)

        y = (y - Y_MEAN) / Y_STD

    return X, y, scales


def create_model(input_shape, output_size=28):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(128, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(output_size))

    return model


def train_detector(train_gt, train_img_dir, fast_train=False):
    input_shape = (*IMG_SHAPE, 1)  # input layer shape
    output_size = len(list(train_gt.values())[0])
    if fast_train:
        keys = list(train_gt.keys())[:100]
        train_gt = {key: train_gt[key] for key in keys}

    X, y, _ = load_data(train_img_dir, train_gt, input_shape, output_size)

    # Model
    epochs = 1 if fast_train else 5000
    learning_rates = np.linspace(0.03, 0.001, epochs)
    patience = 200  # stop if err has not been updated patience time

    # SGD
    lr = 0.01
    momentum = 0.9
    nesterov = True

    change_lr = LearningRateScheduler(lambda epoch:
                                      float(learning_rates[epoch]))
    early_stop = EarlyStopping(patience=patience)

    sgd = SGD(lr=lr, momentum=momentum, nesterov=nesterov)

    model = create_model(input_shape, output_size)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    checkpoint_callback = ModelCheckpoint(filepath='model.hdf5',
                                          monitor='val_loss',
                                          save_best_only=True,
                                          mode='auto')

    start_time = time.time()
    print('start_time: {}'.format(time.strftime('%H:%M:%S')))
    model.fit(X, y,
              epochs=epochs,
              validation_split=0.2,
              callbacks=[checkpoint_callback, change_lr, early_stop])
    print('end_time: {}, duration(min): {}'.format(time.strftime('%H:%M:%S'),
          (time.time()-start_time) / 60.))

    return model


def detect(model, test_img_dir):
    gt = {}
    for fname in glob.glob1(test_img_dir, '*.jpg'):
        gt[fname] = None

    X, _, scales = load_data(test_img_dir, gt, (*IMG_SHAPE, 1), test=True)
    y_pred = model.predict(X)
    y_pred = y_pred * Y_STD + Y_MEAN

    y_scaled = {}
    for i in range(len(scales)):
        scale_x, scale_y, fn = scales[i]
        y = y_pred[i]
        y[::2] = y[::2] * scale_x
        y[1::2] = y[1::2] * scale_y
        y_scaled[fn] = y

    return y_scaled
