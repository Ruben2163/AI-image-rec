#Import Libarys
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

#Creating Batches
data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

train_size = int(len(data)*0.6)
val_size = int(len(data)*0.2)+1
test_size = int(len(data)*0.2)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

#Creating Network
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

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

model.save(os.path.join('models','person_dog.h5'))
