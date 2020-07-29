import glob
import os
import random
import time
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, MeanIoU
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_value
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model

from models.common.common import get_training_gids_from_file, one_hot_encoding, \
    one_hot_to_rgb, split_to_tiles
from models.densenet.densetnet import DenseNet


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
            image_list = split_to_tiles(image / 255, 256)
            images += [image_list[i] for i in indices]

            label = cv2.imread(os.path.join(label_base_dir, f"{gid}.png"), cv2.IMREAD_GRAYSCALE)
            label = one_hot_encoding(label)
            label_list = split_to_tiles(label, 256)
            labels += [label_list[i] for i in indices]

            while len(images) > batch_size:
                image_batch = images[:batch_size]
                label_batch = labels[:batch_size]

                images = images[batch_size:]
                labels = labels[batch_size:]

                yield np.array(image_batch), np.array(label_batch)


def make_training_and_validation_generators(batch_size=4, validation_split=0.1):
    # gids = get_gids_from_database("densenet")
    gids = get_training_gids_from_file("gids_with_multiple_segments.txt")

    rnd = random.Random(42)
    rnd.shuffle(gids)

    split = int(len(gids) * validation_split)

    validation = gids[:split]
    training = gids[split:]

    return data_generator(training, batch_size, seed=17), data_generator(validation, batch_size, seed=29)



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
        images = split_to_tiles(image / 255, 256)

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

        model = DenseNet()
        model.compile(optimizer=Adam(lr=1e-4), loss=weighted_categorical_crossentropy(class_weights), metrics=metrics)
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
