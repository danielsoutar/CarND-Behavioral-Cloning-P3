import csv
import cv2
import numpy as np
import sklearn
import tensorflow as tf
from tqdm import tqdm
from random import shuffle

flags = tf.app.flags
FLAGS = flags.FLAGS

#command-line flags
flags.DEFINE_string('data', '', "File containing trainable data")
flags.DEFINE_string('model', '', "h5 file containing complete model")
flags.DEFINE_integer('num_epochs', 3, "self-explanatory")

lines = []
with open('./' + FLAGS.data + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in tqdm(reader):
        lines.append(line)

images = []
measurements = []
correction = 0.1
for line in tqdm(lines):
    centre_path = line[0]
    left_path = line[1]
    right_path = line[2]

    centre_filename = centre_path.split('/')[-1]
    left_filename = left_path.split('/')[-1]
    right_filename = right_path.split('/')[-1]

    centre_path = './' + FLAGS.data + '/IMG/' + centre_filename
    left_path = './' + FLAGS.data + '/IMG/' + left_filename
    right_path = './' + FLAGS.data + '/IMG/' + right_filename

    centre_image = cv2.imread(centre_path)
    left_image = cv2.imread(left_path)
    right_image = cv2.imread(right_path)
    # convert to RGB
    centre_image = cv2.cvtColor(centre_image, cv2.COLOR_BGR2RGB)
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

    images.append(centre_image)
    images.append(left_image)
    images.append(right_image)

    # add flipped images as well
    images.append(cv2.flip(centre_image, 1))
    images.append(cv2.flip(left_image, 1))
    images.append(cv2.flip(right_image, 1))

    # add a small correction factor to the left and right images
    centre_measurement = float(line[3])
    left_measurement = centre_measurement + correction
    right_measurement = centre_measurement - correction

    measurements.append(centre_measurement)
    measurements.append(left_measurement)
    measurements.append(right_measurement)

    # corresponding to flipped image, add flipped measurements
    measurements.append(centre_measurement*-1.0)
    measurements.append(left_measurement*-1.0)
    measurements.append(right_measurement*-1.0)

print('data retrieved')

X_train = np.array(images)
y_train = np.array(measurements)

from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
# from keras.layers.core import Dropout, Activation
# from keras.layers.pooling import MaxPooling2D
from keras.models import load_model


def construct_model():
    ch, row, col = 3, 160, 320
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2), W_regularizer=l2(0.001)))
    model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2), W_regularizer=l2(0.001)))
    model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2), W_regularizer=l2(0.001)))
    model.add(Convolution2D(64, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution2D(64, 3, 3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Flatten())
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(Dense(1))
    print('model defined...')
    return model


def reconstruct_model(model):
    return load_model(model)

if FLAGS.model == '':
    model = construct_model()
else:
    model = reconstruct_model(FLAGS.model)

model.compile(loss='mse', optimizer=Adam(lr=1e-4))
print('model compiled...')
history_object = model.fit(X_train, y_train, validation_split=0.2, verbose=1, shuffle=True, nb_epoch=FLAGS.num_epochs)

print(history_object.history.keys())

model.save('nvidia-crop-aug-flags-final.h5')
