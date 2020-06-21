import os
import psycopg2
import random
import cv2
import time
import glob

from datetime import datetime

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Cropping2D, concatenate
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, MeanIoU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Model, load_model

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


def data_generator(gids, batch_size, seed=0):
    rnd = random.Random(seed)
    image_base_dir = os.path.join("E:", "data", "unet", "images")
    label_base_dir = os.path.join("E:", "data", "unet", "labels")

    yield len(gids) // batch_size

    while True:
        rnd.shuffle(gids)
        for chunk in chunks(gids, batch_size):
            images, labels = [], []
            for gid in chunk:
                image = cv2.imread(os.path.join(image_base_dir, f"{gid}.png"), cv2.IMREAD_COLOR)
                images.append(image / 255)

                label = cv2.imread(os.path.join(label_base_dir, f"{gid}.png"), cv2.IMREAD_GRAYSCALE)
                labels.append(one_hot_encoding(label))

            # (None, 572, 572, 3)  --  (None, 388, 388, 6)
            yield np.array(images), np.array(labels)


def make_training_and_validation_generators(batch_size=4, validation_split=0.1):
    gids = get_training_gids()

    rnd = random.Random(42)
    rnd.shuffle(gids)

    split = int(len(gids) * validation_split)

    validation = gids[:split]
    training = gids[split:]

    return data_generator(training, batch_size, seed=17), data_generator(validation, batch_size, seed=29)


def one_hot_to_rgb(prediction):
    palette = np.array([(3, 0, 208),  # buildings
                        (240, 126, 11),  # water
                        (40, 171, 44),  # forest
                        (39, 255, 154),  # urban greens
                        (193, 193, 193),  # traffic
                        (132, 240, 235)])  # agriculture

    classes = np.argmax(prediction, axis=2)
    out = np.zeros(classes.shape[:2] + (3,))
    for idx, col in enumerate(palette):
        out[classes == idx] = col
    return out


def predict(gids, weight_path):
    model = define_and_compile_model()
    model.load_weights(weight_path)

    images = []
    for gid in gids:
        image = cv2.imread(os.path.join("E:", "data", "unet", "images", f"{gid}.png"), cv2.IMREAD_COLOR)
        images.append(image / 255)

    pred = model.predict(np.array(images))
    for idx, p in enumerate(pred):
        cv2.imwrite(f"{gids[idx]}-pred.png", one_hot_to_rgb(p))


def do_training(start_time):
    training_gen, validation_gen = make_training_and_validation_generators()
    steps_per_epoch = next(training_gen)
    validation_steps = next(validation_gen)

    if os.path.exists(f"weights/{start_time}"):
        files = glob.glob(f"weights/{start_time}/*.hdf5")
        run = len(files)
        f = max(files, key=os.path.getctime)

        dependencies = {
        }

        model = load_model(f, custom_objects=dependencies)
        # model.summary()
    else:
        run = 0
        os.mkdir(f"weights/{start_time}")

        metrics = [Accuracy(), CategoricalAccuracy(), MeanIoU(num_classes=6)]
        model = define_and_compile_model(metrics=metrics)
        # model.summary()

    logdir = "tf-logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = f"weights/{start_time}/run-{run:02d}__epoch-{{epoch:02d}}__val-loss-{{val_loss:.2f}}.hdf5"

    tensorboard_callback = TensorBoard(log_dir=logdir)
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path)
    logger_callback = CSVLogger(f"weights/{start_time}.csv", append=True)

    model.fit(training_gen, epochs=10, steps_per_epoch=steps_per_epoch,
              validation_data=validation_gen, validation_steps=validation_steps,
              callbacks=[tensorboard_callback, checkpoint_callback, logger_callback])


if __name__ == '__main__':
    # predict([31033, 85616, 156078, 174458], "weights/1592678832/epoch-02__val-loss-0.64.h5")

    start_time = int(time.time())
    do_training(start_time)
