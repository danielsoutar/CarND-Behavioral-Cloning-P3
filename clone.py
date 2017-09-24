import csv
import cv2
import numpy as np
import sklearn
from tqdm import tqdm
from random import shuffle

def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1:
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batches = samples[offset:offset+batch_size]
      
      images = []
      angles = []
      for batch_sample in batches:
        centre_path = './data/IMG/' + batch_sample[0].split('/')[-1]
        centre_image = cv2.imread(centre_path)
        centre_angle = float(batch_sample[3])
        images.append(centre_image)
        angles.append(centre_angle)

      X_train = np.array(images)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

lines = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in tqdm(reader):
    lines.append(line)

from sklearn.model_selection import train_test_split
train_set, validation_set = train_test_split(lines, test_size=0.2)

images = []
measurements = []
correction = 0.1
#for line in tqdm(lines):
#  centre_path = line[0]
#  left_path = line[1]
#  right_path = line[2]

#  centre_filename = centre_path.split('/')[-1]
#  left_filename = left_path.split('/')[-1]
#  right_filename = right_path.split('/')[-1]

#  centre_path = './data/IMG/' + centre_filename
#  left_path = './data/IMG/' + left_filename
#  right_path = './data/IMG/' + right_filename

#  centre_image = cv2.imread(centre_path)
#  left_image = cv2.imread(left_path)
#  right_image = cv2.imread(right_path)
#  images.extend((centre_image, left_image, right_image))
#  images.append(centre_image)

#  centre_measurement = float(line[3])
#  left_measurement = centre_measurement + correction
#  right_measurement = centre_measurement - correction

#  measurements.extend((centre_measurement, left_measurement, right_measurement))
#  measurements.append(centre_measurement)
#print('data retrieved')

#augmented_images, augmented_measurements = [], []
#for image, measurement in zip(images, measurements):
#  augmented_images.append(image)
#  images.remove(image)
#  augmented_measurements.append(measurement)
#  measurements.remove(measurement)
#  augmented_images.append(cv2.flip(image,1))
#  augmented_measurements.append(measurement*-1.0)

#print('data augmented')

#X_train = np.array(augmented_images)
#y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.core import Dropout, Activation
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
import matplotlib.pyplot as plt

ch, row, col = 3, 160, 320

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row,col,ch), output_shape=(row, col, ch)))
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

print('model defined...')

train_generator = generator(train_set, batch_size=128)
validation_generator = generator(validation_set, batch_size=128)

print('testing generator...')

gen = generator(train_set)
my_output = (next(gen))

model.compile(loss='mse', optimizer='adam')
print('model compiled...')
history_object = model.fit_generator(train_generator, \
  samples_per_epoch = len(train_set), \
  validation_data = validation_generator, \
  nb_val_samples=len(validation_set), \
  nb_epoch=5, verbose=1)

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mse loss')
plt.ylabel('mse loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('lenet-crop-aug.h5')
