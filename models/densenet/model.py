import os
import psycopg2
import random
import cv2
import time
import glob

from datetime import datetime

from tensorflow.keras.layers import Input, Activation, Conv2D, Dropout, BatchNormalization, AveragePooling2D, \
    Concatenate, Conv2DTranspose
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, MeanIoU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.backend import set_value

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils.vis_utils import plot_model

from models.common.common import get_training_gids_from_database, get_training_gids_from_file, one_hot_encoding


def conv_block(x, nb_filters, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    if bottleneck:
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
        x = Activation("relu")(x)
        x = Conv2D(nb_filters * 4, (1, 1), padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = Conv2D(nb_filters, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def dense_block(x, nb_layers, nb_filters, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
    for _ in range(nb_layers):
        block = conv_block(x, growth_rate, dropout_rate, bottleneck, weight_decay)
        x = Concatenate(axis=3)([x, block])
        nb_filters += growth_rate

    return x, nb_filters


def transition_down_layer(x, nb_filters, dropout_rate=None, compression=1.0, weight_decay=1e-4):
    nb_filters = int(nb_filters * compression)
    x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = Conv2D(nb_filters, (1, 1), padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x, nb_filters


def transition_up_layer(skip_connection, x, nb_filters):
    x = Conv2DTranspose(nb_filters, (2, 2), strides=(2, 2))(x)
    x = Concatenate(axis=3)([x, skip_connection])
    return x, nb_filters


def define_and_compile_model(optimizer=Adam(lr=1e-4), loss=None, metrics=None):
    input_size = (256, 256, 3)
    growth_rate = 16
    weight_decay = 1e-4
    dense_block_layers = [4, 5, 7, 10, 12, 15]
    dropout = 0.2
    compression = 1.0
    bottleneck = True
    nb_filters = 48
    skip_connections = []

    input = Input(input_size)
    x = Conv2D(nb_filters, (3, 3), padding="same", use_bias=False, kernel_regularizer=l2(weight_decay))(input)

    for i, block_size in enumerate(dense_block_layers):
        x, nb_filters = dense_block(x, block_size, nb_filters, growth_rate, dropout, bottleneck, weight_decay)

        skip_connections.append(x)
        if i < len(dense_block_layers) - 1:
            x, nb_filters = transition_down_layer(x, nb_filters, dropout, compression, weight_decay)

    skip_connections = skip_connections[::-1][1:]

    for i, block_size in enumerate(dense_block_layers[::-1][1:]):
        x, nb_filters = transition_up_layer(skip_connections[i], x, nb_filters)

        x, nb_filters = dense_block(x, block_size, nb_filters, growth_rate, dropout, bottleneck, weight_decay)

    x = Conv2D(6, (1, 1), activation="softmax", kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(
        x)

    model = Model(inputs=input, outputs=x)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model




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
    label_base_dir = os.path.join("E:", "data", "densenet", "train", "labels")

    images_per_file = 25
    yield (len(gids) // batch_size) * images_per_file

    while True:
        rnd.shuffle(gids)
        images, labels = [], []

        for gid in gids:
            indices = random.sample(range(100), 25)

            image = cv2.imread(os.path.join(image_base_dir, f"{gid}.png"), cv2.IMREAD_COLOR)
            image_list = split_to_tiles(image / 255)
            images += [image_list[i] for i in indices]

            label = cv2.imread(os.path.join(label_base_dir, f"{gid}.png"), cv2.IMREAD_GRAYSCALE)
            label = one_hot_encoding(label)
            label_list = split_to_tiles(label)
            labels += [label_list[i] for i in indices]

            while len(images) > batch_size:
                image_batch = images[:batch_size]
                label_batch = labels[:batch_size]

                images = images[batch_size:]
                labels = labels[batch_size:]

                yield np.array(image_batch), np.array(label_batch)


def make_training_and_validation_generators(batch_size=4, validation_split=0.1):
    # gids = get_training_gids_from_database("densenet")
    gids = get_training_gids_from_file("gids_with_multiple_segments.txt")

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


def weighted_categorical_crossentropy(class_weights):
    class_weights = tf.constant(class_weights)

    def wcce(y_true, y_pred):
        # weights = tf.reduce_sum(tf.multiply(y_true, class_weights), axis=1)
        # return tf.compat.v1.losses.softmax_cross_entropy(y_true, y_pred, weights=weights)

        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=class_weights)

    return wcce


def predict(gids, model_path):
    dependencies = {
        'wcce': weighted_categorical_crossentropy(class_weights=[
            0.02471,  # buildings
            0.54651,  # water
            0.01185,  # forest
            1.00000,  # traffic
            0.16433,  # urban greens
            0.01470,  # agriculture
        ])
    }
    model = load_model(model_path, custom_objects=dependencies)

    for gid in gids:
        image = cv2.imread(os.path.join("E:", "data", "densenet", "train", "images", f"{gid}.png"), cv2.IMREAD_COLOR)
        images = split_to_tiles(image / 255)

        final_prediction = np.empty([0, 2560, 3])
        row = np.empty([256, 0, 3])

        for idx, img in enumerate(images):
            pred = model.predict(np.array([images[idx]]))
            row = np.hstack([row, one_hot_to_rgb(pred[0])])
            if row.shape[1] >= 2560:
                final_prediction = np.vstack([final_prediction, row])
                row = np.empty([256, 0, 3])

        cv2.imwrite(f"images/{gid}-pred.png", final_prediction)


def do_training(start_time):
    training_gen, validation_gen = make_training_and_validation_generators(batch_size=1)
    steps_per_epoch = next(training_gen)
    validation_steps = next(validation_gen)

    # weight the classes according to the area they cover in the dataset
    class_weights = [
        0.02471,  # buildings
        0.54651,  # water
        0.01185,  # forest
        1.00000,  # traffic
        0.16433,  # urban greens
        0.01470,  # agriculture
    ]

    if os.path.exists(f"weights/{start_time}"):
        files = glob.glob(f"weights/{start_time}/*.hdf5")
        run = len(files)
        f = max(files, key=os.path.getctime)
        f = "weights/1594878370/run-00__epoch-11__val-loss-0.64.hdf5"

        dependencies = {
            'wcce': weighted_categorical_crossentropy(class_weights)
        }

        model = load_model(f, custom_objects=dependencies)
        set_value(model.optimizer.lr, 1e-5)
        model.summary()
    else:
        run = 0
        os.mkdir(f"weights/{start_time}")

        metrics = [Accuracy(), CategoricalAccuracy(), MeanIoU(num_classes=6)]

        model = define_and_compile_model(metrics=metrics, loss=weighted_categorical_crossentropy(class_weights))
        model.summary()
        plot_model(model, to_file=f"{start_time}.png")

    logdir = "tf-logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = f"weights/{start_time}/run-{run:02d}__epoch-{{epoch:02d}}__val-loss-{{val_loss:.2f}}.hdf5"

    tensorboard_callback = TensorBoard(log_dir=logdir)
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path)
    logger_callback = CSVLogger(f"weights/{start_time}.csv", append=True)

    model.fit(training_gen, epochs=50, steps_per_epoch=steps_per_epoch,
              validation_data=validation_gen, validation_steps=validation_steps,
              callbacks=[tensorboard_callback, checkpoint_callback, logger_callback])


if __name__ == '__main__':
   # predict([44], "weights/1593860558/run-00__epoch-02__val-loss-1.70.hdf5")
   # exit(0)

    start_time = int(time.time())
    do_training(start_time)
