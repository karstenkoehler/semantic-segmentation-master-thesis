import os
import time
import random
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy
from tensorflow.keras.metrics import CategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.losses import categorical_crossentropy

from models.common.callbacks import metrics_to_csv_logger, save_model_on_epoch_end
from models.common.common import get_gids_from_database, one_hot_encoding, one_hot_to_rgb
from models.common.data_generator import initialize_train_and_validation_generators
from models.common.metrics import TF_CUSTOM_METRICS
from models.wnet.wnet import WNet


def predict(gids, model_path, mode="train"):
    model = load_model(model_path, TF_CUSTOM_METRICS)

    images = []
    labels = []
    for gid in gids:
        image = cv2.imread(os.path.join("E:", "data", "wnet", mode, "images", f"{gid}.png"), cv2.IMREAD_COLOR)
        images.append(image / 255)

        label = cv2.imread(os.path.join("E:", "data", "wnet", mode, "labels", f"{gid}.png"), cv2.IMREAD_GRAYSCALE)
        labels.append(one_hot_encoding(label))

    pred = model.predict(np.array(images))
    for idx, p in enumerate(pred):
        cv2.imwrite(f"images/{mode}/{gids[idx]}-prediction.png", one_hot_to_rgb(p))
        cv2.imwrite(f"images/{mode}/{gids[idx]}-label.png", one_hot_to_rgb(labels[idx]))
        cv2.imwrite(f"images/{mode}/{gids[idx]}-image.png", images[idx] * 255)


def restore(gids, model_path, mode="train"):
    model = load_model(model_path, TF_CUSTOM_METRICS)

    images = []
    for gid in gids:
        image = cv2.imread(os.path.join("E:", "data", "wnet", mode, "images", f"{gid}.png"), cv2.IMREAD_COLOR)
        images.append(image / 255)

    pred = model.predict(np.array(images))

    for idx, p in enumerate(pred):
        cv2.imwrite(f"images/{mode}/{gids[idx]}-restored.png", p * 255)


def do_training(initial_learning_rate=0.001):
    gids = get_gids_from_database("wnet")
    training_gen, validation_gen = initialize_train_and_validation_generators("wnet", gids, batch_size=5,
                                                                              label_target_size=256,
                                                                              use_image_as_label=True)
    steps_per_epoch = next(training_gen)
    validation_steps = next(validation_gen)

    full_model, encoder_model = WNet(nb_classes=6)
    metrics = [Accuracy(), CategoricalAccuracy(), CategoricalCrossentropy()]
    optimizer = Adam(lr=initial_learning_rate)
    full_model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=metrics)

    start_time = int(time.time())
    model_path = f"weights/{start_time}_{full_model.name}/"
    os.mkdir(model_path)

    metrics_to_log = [metric.name for metric in metrics]
    callbacks = [
        save_model_on_epoch_end(full_model.name, full_model, model_path),
        save_model_on_epoch_end(encoder_model.name, encoder_model, model_path),
        metrics_to_csv_logger(model_path + "/batch.csv", ["loss"] + metrics_to_log),
        CSVLogger(model_path + "/epoch.csv", separator=";"),
    ]
    full_model.fit(training_gen, epochs=1, steps_per_epoch=steps_per_epoch,
                   validation_data=validation_gen, validation_steps=validation_steps,
                   callbacks=callbacks)


if __name__ == '__main__':
    np.random.seed(1595840929)
    random.seed(1595840929)
    tf.random.set_seed(1595840929)

    # gids = [63585, 224817, 19743, 78905, 102574, 119756]
    # predict(gids, "weights/1596694200_WNet-46D-6/WNet-46D-6-Encoder_epoch_9.hdf5", mode="test")
    # restore(gids, "weights/1596694200_WNet-46D-6/WNet-46D-6_epoch_9.hdf5", mode="test")
    # exit(0)

    do_training()
