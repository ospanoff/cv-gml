import time
import glob
import numpy as np
import skimage.io as skio
import skimage.transform as sktr

from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, Activation, BatchNormalization
)
from keras.optimizers import SGD
from keras.callbacks import (
    LearningRateScheduler, EarlyStopping, ModelCheckpoint
)


Y_MEAN = 50
Y_STD = 50
IMG_SHAPE = (100, 100)


def load_data(img_dir, gt, input_shape, output_size=28, test=False):
    X = np.empty((len(gt), *input_shape))
    y = np.empty((len(gt), output_size)) if not test else None
    scales = []

    for i, (fn, y_) in enumerate(gt.items()):
        img = skio.imread(img_dir + '/' + fn)
        scale_y = 1.0 * img.shape[0] / input_shape[0]
        scale_x = 1.0 * img.shape[1] / input_shape[1]
        scales += [(scale_x, scale_y, fn)]

        X[i] = sktr.resize(img, input_shape)

        if not test:
            y_[::2] = y_[::2] / scale_x
            y_[1::2] = y_[1::2] / scale_y
            y[i] = y_

    if not test:
        X_fl = X[:, :, ::-1].copy()
        y_fl = y.copy()
        y_fl[:, ::2] = input_shape[1] - y_fl[:, ::2] - 1

        X = np.vstack((X, X_fl))
        y = np.vstack((y, y_fl))

        y = (y - Y_MEAN) / Y_STD

    return X, y, scales


def create_model(input_shape, output_size=28):
    model = Sequential()

    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(32, (3, 3)))  # , input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(output_size))

    return model


def train_detector(train_gt, train_img_dir, fast_train=False):
    input_shape = (*IMG_SHAPE, 3)  # input layer shape
    output_size = len(list(train_gt.values())[0])
    if fast_train:
        keys = list(train_gt.keys())[:10]
        train_gt = {key: train_gt[key] for key in keys}

    X, y, _ = load_data(train_img_dir, train_gt, input_shape, output_size)

    # Model config.
    epochs = 1 if fast_train else 500
    learning_rates = np.linspace(0.03, 0.0001, epochs)
    patience = 100  # stop if err has not been updated patience time

    change_lr = LearningRateScheduler(lambda epoch: learning_rates[epoch])
    early_stop = EarlyStopping(patience=patience)

    # SGD config.
    lr = 0.03
    momentum = 0.9
    sgd = SGD(lr=lr, momentum=momentum, nesterov=True)

    # Model setup
    model = create_model(input_shape, output_size)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    checkpoint_callback = ModelCheckpoint(filepath='model.hdf5',
                                          monitor='val_loss',
                                          save_best_only=True,
                                          mode='auto')

    # Model training
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

    X, _, scales = load_data(test_img_dir, gt, (*IMG_SHAPE, 3), test=True)
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
