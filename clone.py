import csv
import cv2
import numpy as np
from tqdm import tqdm

lines = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
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

  centre_path = './data/IMG/' + centre_filename
  left_path = './data/IMG/' + left_filename
  right_path = './data/IMG/' + right_filename

  centre_image = cv2.imread(centre_path)
  left_image = cv2.imread(left_path)
  right_image = cv2.imread(right_path)
  images.extend(centre_image, left_image, right_image)

  centre_measurement = float(line[3])
  left_measurement = centre_measurement + correction
  right_measurement = centre_measurement - correction

  measurements.extend(centre_measurement, left_measurement, right_measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image,1))
  augmented_measurements.append(measurements*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print('data retrieved')

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,2,2,activation='relu', W_regularizer=l2(0.01)))
model.add(MaxPooling2D())
model.add(Convolution2D(48,2,2,activation='relu', W_regularizer=l2(0.01)))
model.add(MaxPooling2D())
model.add(Convolution2D(96,2,2,activation='relu', W_regularizer=l2(0.01)))
model.add(MaxPooling2D())
model.add(Convolution2D(192,2,2,activation='relu', W_regularizer=l2(0.01)))
model.add(Flatten())
model.add(Dense(2000, W_regularizer=l2(0.01)))
model.add(Dropout(0.9))
model.add(Activation('relu'))
model.add(Dense(500, W_regularizer=l2(0.01)))
model.add(Dropout(0.8))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('lenet-crop-aug.h5')
