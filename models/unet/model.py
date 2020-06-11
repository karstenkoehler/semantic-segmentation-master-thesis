from datetime import datetime

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Cropping2D, concatenate
from keras.metrics import Accuracy, TopKCategoricalAccuracy, MeanIoU
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.models import Model

import tensorflow as tf

# contracting path
input_size = (572, 572, 3)
input = Input(input_size)
conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(input)
conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool1)
conv2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool2)
conv3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool3)
conv4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

# expansive path
up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(drop5))
merge6 = concatenate([Cropping2D(4)(drop4), up6], axis=3)
conv6 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge6)
conv6 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv6)

up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv6))
merge7 = concatenate([Cropping2D(16)(conv3), up7], axis=3)
conv7 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge7)
conv7 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv7)

up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv7))
merge8 = concatenate([Cropping2D(40)(conv2), up8], axis=3)
conv8 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge8)
conv8 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv8)

up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
merge9 = concatenate([Cropping2D(88)(conv1), up9], axis=3)
conv9 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge9)
conv9 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv9)
conv10 = Conv2D(6, 1, activation='sigmoid')(conv9)

model = Model(inputs=input, outputs=conv10)
metrics = [Accuracy(), TopKCategoricalAccuracy(k=2), MeanIoU(num_classes=6)]
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=metrics)
model.summary()

logdir = "tf-logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)

# TODO: provide training data
train_images = tf.fill([3, 572, 572, 3], 9)
train_labels = tf.fill([3, 388, 388, 6], 0.5)
model.fit(train_images, train_labels, epochs=1, steps_per_epoch=1, callbacks=[tensorboard_callback])
