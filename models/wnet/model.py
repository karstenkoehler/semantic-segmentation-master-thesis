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
from tensorflow.python.keras.backend import set_value

import numpy as np


def define_and_compile_model(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=None):
    # contracting path
    input_size = (256, 256, 3)
    input = Input(input_size)
    classification = unet(input, num_classes=6)
    reconstructed = unet(classification, num_classes=3)

    model = Model(inputs=input, outputs=reconstructed)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    encoder = Model(inputs=input, outputs=classification)
    encoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model, encoder


def unet(input, num_classes):
    conv1 = Conv2D(64, 3, activation='relu', padding="same", kernel_initializer='he_normal')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding="same", kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding="same", kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding="same", kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(1024, 3, activation='relu', padding="same", kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    # expansive path
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding="same", kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding="same", kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding="same", kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding="same", kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding="same", kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)
    return conv10


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


def get_training_gids_only_multisegment():
    # returns only the tiles that contain at least two different segments
    with open("gids_with_multiple_segments.txt", 'r') as f:
        gids = [int(line) for line in f.read().splitlines()]
        return gids


def one_hot_encoding(label):
    encoded = []
    for val in [62, 104, 118, 193, 200, 226]:
        encoded.append((label == val) * 1.0)
    return np.stack(encoded, axis=2)


def split_to_tiles(img, tile_size=256):
    tiles = []
    steps = img.shape[0] // tile_size

    for x in range(steps):
        for y in range(steps):
            tile = img[x * tile_size:(x + 1) * tile_size, y * tile_size:(y + 1) * tile_size, :]
            tiles.append(tile)

    return tiles


def data_generator(gids, batch_size, seed=0):
    rnd = random.Random(seed)
    image_base_dir = os.path.join("E:", "data", "densenet", "train", "images")

    images_per_file = 100
    yield (len(gids) // batch_size) * images_per_file

    while True:
        rnd.shuffle(gids)
        images = []

        for gid in gids:
            image = cv2.imread(os.path.join(image_base_dir, f"{gid}.png"), cv2.IMREAD_COLOR)
            images += split_to_tiles(image / 255)

            while len(images) > batch_size:
                image_batch = images[:batch_size]
                images = images[batch_size:]
                yield np.array(image_batch), np.array(image_batch)


def make_training_and_validation_generators(batch_size=4, validation_split=0.1):
    # gids = get_training_gids()
    gids = get_training_gids_only_multisegment()

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
                        (193, 193, 193),  # traffic
                        (39, 255, 154),  # urban greens
                        (132, 240, 235)])  # agriculture

    classes = np.argmax(prediction, axis=2)
    out = np.zeros(classes.shape[:2] + (3,))
    for idx, col in enumerate(palette):
        out[classes == idx] = col
    return out


def predict(gids, model_path):
    model = load_model(model_path)

    images = []
    for gid in gids:
        image = cv2.imread(os.path.join("E:", "data", "wnet", "images", f"{gid}.png"), cv2.IMREAD_COLOR)
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
        f = "weights/1593163475/run-00__epoch-38__val-loss-0.94.hdf5"

        dependencies = {
        }

        model = load_model(f, custom_objects=dependencies)
        set_value(model.optimizer.lr, 1e-5)
        model.summary()
    else:
        run = 0
        os.mkdir(f"weights/{start_time}")

        metrics = [Accuracy(), CategoricalAccuracy()]
        model, encoder = define_and_compile_model(metrics=metrics)
        model.summary()
        encoder.summary()

    logdir = "tf-logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = f"weights/{start_time}/run-{run:02d}__epoch-{{epoch:02d}}__val-loss-{{val_loss:.2f}}.hdf5"

    tensorboard_callback = TensorBoard(log_dir=logdir)
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path)
    logger_callback = CSVLogger(f"weights/{start_time}.csv", append=True)

    # TODO: save model and encoder separately. or find a way to use model weights to init the encoder
    model.fit(training_gen, epochs=50, steps_per_epoch=steps_per_epoch,
              validation_data=validation_gen, validation_steps=validation_steps,
              callbacks=[tensorboard_callback, checkpoint_callback, logger_callback])


if __name__ == '__main__':
    # predict([285, 304, 345, 14390, 31033, 85616, 156078, 174458], "weights/1593163475/run-00__epoch-50__val-loss-1.07.hdf5")
    # exit(0)

    start_time = int(time.time())
    do_training(start_time)
