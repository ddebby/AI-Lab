from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import load_model

import os
import numpy as np

# add tensorboard callback
log_dir="./logs/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(
                        log_dir=log_dir, histogram_freq=1)

# checkpoint
filepath="model01-{epoch:02d}-{val_accuracy:.2f}.hdf5"
# 只保存最好的模型 只需要将文件名改成固定的（新的好的覆盖旧的） filepath="weights.best.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
mode='max')

#Early Stopping
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

#Configs
img_dir = "/home/inesa-gao/.fastai/data/cifar10/train"
img_size = (32,32)
batch_size=32
epochs = 30


image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                 directory=img_dir,
                                                 shuffle=True,
                                                 target_size=img_size,
                                                 subset="training",
                                                 class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=batch_size,
                                                 directory=img_dir,
                                                 shuffle=True,
                                                 target_size=img_size,
                                                 subset="validation",
                                                 class_mode='categorical')
num_class = train_dataset.num_classes


model = Sequential([
    Conv2D(16, (3, 3), padding='same', input_shape=train_dataset.image_shape),
    Activation('relu'),
    Conv2D(16, (3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.25),

    Conv2D(32, (3, 3), padding='same'),
    Activation('relu'),
    Conv2D(32, (3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    Dropout(0.25),

    Conv2D(128, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    GlobalAveragePooling2D(),
    #    Dropout(0.2),
    Dense(num_class, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=1e-3, decay=1e-6),
              metrics=['accuracy'])

model.fit_generator(
        train_dataset,
        steps_per_epoch=train_dataset.n // batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=validation_dataset,
        #validation_steps=800,
        callbacks=[tensorboard_callback, checkpoint, es])

model.save('model.h5')