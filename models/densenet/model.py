import os
import os
import random
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy
from tensorflow.keras.models import load_model
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import CategoricalCrossentropy
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp

from models.common.callbacks import save_model_on_epoch_end, metrics_to_csv_logger
from models.common.common import one_hot_to_rgb, split_to_tiles, get_gids_from_database
from models.common.data_generator import initialize_train_and_validation_generators
from models.common.metrics import ArgmaxMeanIoU, weighted_categorical_crossentropy
from models.densenet.densenet import DenseNet


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


def lr_schedule(initial_lr=0.01, factor=5, power=2):
    def schedule(epoch):
        return initial_lr / (factor * (epoch + 1) ** power)

    return schedule


def do_training(initial_learning_rate=0.001):
    gids = get_gids_from_database("densenet")
    training_gen, validation_gen = initialize_train_and_validation_generators("densenet", gids, batch_size=4, label_target_size=224)
    steps_per_epoch = next(training_gen)
    validation_steps = next(validation_gen)

    model = DenseNet()
    metrics = [Accuracy(), CategoricalAccuracy(),
               CategoricalCrossentropy(), ArgmaxMeanIoU(num_classes=6, name="mean_iou")]
    optimizer = RMSProp(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=metrics)

    start_time = int(time.time())
    model_path = f"weights/{start_time}_{model.name}/"
    os.mkdir(model_path)

    metrics_to_log = [metric.name for metric in metrics]
    callbacks = [
        save_model_on_epoch_end(model.name, model, model_path),
        metrics_to_csv_logger(model_path + "/batch.csv", ["loss"] + metrics_to_log),
        CSVLogger(model_path + "/epoch.csv", separator=";"),
        LearningRateScheduler(lr_schedule(initial_lr=initial_learning_rate)),
    ]
    model.fit(training_gen, epochs=50, steps_per_epoch=steps_per_epoch,
              validation_data=validation_gen, validation_steps=validation_steps,
              callbacks=callbacks)


if __name__ == '__main__':
    np.random.seed(1595840929)
    random.seed(1595840929)
    tf.random.set_seed(1595840929)

    # predict([44], "weights/1593860558/run-00__epoch-02__val-loss-1.70.hdf5")
    # exit(0)

    do_training()
