import os
import time

import cv2
import numpy as np
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, MeanIoU
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import set_value

from models.common.callbacks import metrics_to_csv_logger, save_model_on_epoch_end
from models.common.common import get_gids_from_database, one_hot_to_rgb
from models.common.data_generator import initialize_train_and_validation_generators
from models.unet.unet import UNet


def define_and_compile_model(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=None):
    if metrics is None:
        metrics = [Accuracy(), CategoricalAccuracy(), MeanIoU(num_classes=6)]

    model, _ = UNet()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def predict(gids, model_path):
    model = load_model(model_path)

    images = []
    for gid in gids:
        image = cv2.imread(os.path.join("E:", "data", "unet", "images", f"{gid}.png"), cv2.IMREAD_COLOR)
        images.append(image / 255)

    pred = model.predict(np.array(images))
    for idx, p in enumerate(pred):
        cv2.imwrite(f"{gids[idx]}-pred.png", one_hot_to_rgb(p))


def do_training(continue_from_file=""):
    gids = get_gids_from_database("unet")
    training_gen, validation_gen = initialize_train_and_validation_generators("unet", gids, batch_size=4)
    steps_per_epoch = next(training_gen)
    validation_steps = next(validation_gen)

    if continue_from_file != "" and os.path.exists(continue_from_file):
        model = load_model(continue_from_file)
        set_value(model.optimizer.lr, 1e-5)
    else:
        model = define_and_compile_model()

        start_time = int(time.time())
        os.mkdir(f"weights/{start_time}_{model.name}/")

    metrics_to_log = ["loss", "accuracy", "categorical_accuracy", "mean_io_u"]
    callbacks = [
        save_model_on_epoch_end(model.name, model, f"weights/{start_time}_{model.name}/"),
        metrics_to_csv_logger(f"weights/{start_time}_{model.name}.csv", metrics_to_log),
    ]

    model.fit(training_gen, epochs=50, steps_per_epoch=steps_per_epoch,
              validation_data=validation_gen, validation_steps=validation_steps,
              callbacks=callbacks)


if __name__ == '__main__':
    # predict([285, 304, 345, 14390, 31033, 85616, 156078, 174458], "weights/1593163475/run-00__epoch-50__val-loss-1.07.hdf5")
    # exit(0)

    do_training()
