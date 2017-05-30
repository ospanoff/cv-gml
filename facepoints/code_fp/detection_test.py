#!/usr/bin/python3

from detection import train_detector, detect
from keras.models import load_model
from numpy import array
from os.path import join
from sys import argv, exit

if len(argv) != 3:
    print('Usage: %s train_dir test_dir' % argv[0])
    exit(0)

train_dir = argv[1]
test_dir = argv[2]


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res


def read_img_shapes(gt_dir):
    img_shapes = {}
    with open(join(gt_dir, 'img_shapes.csv')) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            filename = parts[0]
            n_rows, n_cols = map(int, parts[1:])
            img_shapes[filename] = (n_rows, n_cols)
    return img_shapes


def compute_metric(detected, gt, img_shapes):
    res = 0.0
    for filename, coords in detected.items():
        n_rows, n_cols = img_shapes[filename]
        diff = (coords - gt[filename])
        diff[::2] /= n_cols
        diff[1::2] /= n_rows
        diff *= 100
        res += (diff ** 2).mean()
    return res / len(detected.keys())


train_gt = read_csv(join(train_dir, 'gt.csv'))
train_img_dir = join(train_dir, 'images')

model = train_detector(train_gt, train_img_dir, fast_train=False)
model.save('facepoints_model.hdf5')

# train_detector(train_gt, train_img_dir, fast_train=True)

# model = load_model('facepoints_model.hdf5')
# model = load_model('model.hdf5')
test_img_dir = join(test_dir, 'images')
detected_points = detect(model, test_img_dir)

test_gt = read_csv(join(test_dir, 'gt.csv'))
img_shapes = read_img_shapes(test_dir)
error = compute_metric(detected_points, test_gt, img_shapes)
print('Error: ', error)
