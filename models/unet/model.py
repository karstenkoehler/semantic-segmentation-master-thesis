import os
import cv2
import time
import glob

from datetime import datetime

from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, MeanIoU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_value

import numpy as np

from models.common.common import get_training_gids_from_database, one_hot_to_rgb
from models.common.data_generator import initialize_train_and_validation_generators
from models.unet.unet import UNet


def define_and_compile_model(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=None):
    model = UNet()
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


def do_training(start_time):
    gids = get_training_gids_from_database("unet")
    training_gen, validation_gen = initialize_train_and_validation_generators("unet", gids, batch_size=4)
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

    model.fit(training_gen, epochs=50, steps_per_epoch=steps_per_epoch,
              validation_data=validation_gen, validation_steps=validation_steps,
              callbacks=[tensorboard_callback, checkpoint_callback, logger_callback])


if __name__ == '__main__':
    # predict([285, 304, 345, 14390, 31033, 85616, 156078, 174458], "weights/1593163475/run-00__epoch-50__val-loss-1.07.hdf5")
    # exit(0)

    start_time = int(time.time())
    do_training(start_time)
