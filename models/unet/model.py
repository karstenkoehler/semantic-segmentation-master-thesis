import os
import psycopg2
import random
import cv2

from datetime import datetime

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Cropping2D, concatenate
from keras.metrics import Accuracy, TopKCategoricalAccuracy, MeanIoU
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.models import Model

import numpy as np


def define_and_compile_model(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=None):
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

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([Cropping2D(88)(conv1), up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(6, 1, activation='sigmoid')(conv9)

    model = Model(inputs=input, outputs=conv10)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def chunks(gids, n):
    for i in range(0, len(gids), n):
        yield gids[i:i + n]


def get_training_gids():
    db_connection = "dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'"
    with psycopg2.connect(db_connection) as db:
        with db.cursor() as cur:
            stmt = "SELECT gid FROM geom_tiles_unet WHERE NOT test_set;;"
            cur.execute(stmt)
            return [int(row[0]) for row in cur.fetchall()]


def one_hot_encoding(label):
    encoded = []
    for val in [62, 104, 118, 193, 200, 226]:
        encoded.append((label == val) * 1.0)
    return np.stack(encoded, axis=2)


def data_generator(batch_size=2):
    random.seed(42)
    image_base_dir = os.path.join("E:", "data", "unet", "images")
    label_base_dir = os.path.join("E:", "data", "unet", "labels")
    gids = get_training_gids()

    # number of batches per epoch
    yield len(gids) // batch_size

    while True:
        random.shuffle(gids)
        for chunk in chunks(gids, batch_size):
            images, labels = [], []
            for gid in chunk:
                image = cv2.imread(os.path.join(image_base_dir, f"{gid}.png"), cv2.IMREAD_COLOR)
                images.append(image / 255)

                label = cv2.imread(os.path.join(label_base_dir, f"{gid}.png"), cv2.IMREAD_GRAYSCALE)
                labels.append(one_hot_encoding(label))

            # (None, 572, 572, 3)  --  (None, 388, 388, 6)
            yield np.array(images), np.array(labels)


if __name__ == '__main__':
    gen = data_generator()
    steps_per_epoch = next(gen)

    metrics = [Accuracy(), MeanIoU(num_classes=6)]
    model = define_and_compile_model(metrics=metrics)
    # model.summary()

    logdir = "tf-logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    model.fit(gen, epochs=1, steps_per_epoch=steps_per_epoch, callbacks=[tensorboard_callback])
