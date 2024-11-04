from datetime import datetime
StartTime = datetime.now()

import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os



Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout
load_model = tf.keras.models.load_model


data = tf.keras.utils.image_dataset_from_directory('TrainingBenchMark/TestData')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

train_size = int(len(data)*0.6)
val_size = int(len(data)*0.2)+1
test_size = int(len(data)*0.2)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3 ,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(15, (3 ,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

Logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=Logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

model.save('Person_Dog_Rec.h5')

print(f'BenchMark Time was:{(datetime.now() - StartTime)}')