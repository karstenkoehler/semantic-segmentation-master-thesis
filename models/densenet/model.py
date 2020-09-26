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
from models.common.common import one_hot_to_rgb, get_gids_from_database, one_hot_encoding
from models.common.data_generator import initialize_train_and_validation_generators
from models.common.metrics import ArgmaxMeanIoU, TF_CUSTOM_METRICS
from models.densenet.densenet import DenseNet


def predict(gids, model_path, mode="train"):
    model = load_model(model_path, custom_objects=TF_CUSTOM_METRICS)

    images = []
    labels = []
    for gid in gids:
        image = cv2.imread(os.path.join("E:", "data", "densenet", mode, "images", f"{gid}.png"), cv2.IMREAD_COLOR)
        images.append(image / 255)

        label = cv2.imread(os.path.join("E:", "data", "densenet", mode, "labels", f"{gid}.png"), cv2.IMREAD_GRAYSCALE)
        labels.append(one_hot_encoding(label))

    pred = model.predict(np.array(images))
    losses = categorical_crossentropy(labels, pred)
    losses = np.mean(losses, axis=(1, 2))

    argmax_mean_iou = ArgmaxMeanIoU(num_classes=6)
    for idx, p in enumerate(pred):
        argmax_mean_iou.update_state(labels[idx], p)
        iou = argmax_mean_iou.result().numpy()

        print(f"{gids[idx]}: loss={losses[idx]:02f}     iou={iou:02f}")

        cv2.imwrite(f"images/{mode}/{gids[idx]}-prediction.png", one_hot_to_rgb(p))
        cv2.imwrite(f"images/{mode}/{gids[idx]}-label.png", one_hot_to_rgb(labels[idx]))
        cv2.imwrite(f"images/{mode}/{gids[idx]}-image.png", images[idx] * 255)



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

    # predict([1278], "weights/1596568933_FC-DenseNet-67D/FC-DenseNet-67D_epoch_19.hdf5", mode="test")
    # exit(0)

    do_training()
