import os
import time

import cv2
import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, CategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import LearningRateScheduler, CSVLogger

from models.common.callbacks import metrics_to_csv_logger, save_model_on_epoch_end
from models.common.common import get_gids_from_database, one_hot_to_rgb, one_hot_encoding
from models.common.data_generator import initialize_train_and_validation_generators
from models.common.metrics import ArgmaxMeanIoU, TF_CUSTOM_METRICS, weighted_categorical_crossentropy
from models.unet.unet import UNet


def predict(gids, model_path, mode="train"):
    model = load_model(model_path, custom_objects=TF_CUSTOM_METRICS)

    images = []
    labels = []
    for gid in gids:
        image = cv2.imread(os.path.join("E:", "data", "unet", mode, "images", f"{gid}.png"), cv2.IMREAD_COLOR)
        images.append(image / 255)

        label = cv2.imread(os.path.join("E:", "data", "unet", mode, "labels", f"{gid}.png"), cv2.IMREAD_GRAYSCALE)
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


def do_training(initial_learning_rate=0.1):
    gids = get_gids_from_database("unet")
    training_gen, validation_gen = initialize_train_and_validation_generators("unet", gids, batch_size=4, label_target_size=388)
    steps_per_epoch = next(training_gen)
    validation_steps = next(validation_gen)

    class_weights = [
        0.0813,  # buildings
        0.5272,  # water
        0.0482,  # forest
        1.0000,  # traffic
        0.4133,  # urban greens
        0.0480,  # agriculture
    ]

    model, _ = UNet(input_size=(572, 572, 3), model_name_suffix="-CW")
    metrics = [Accuracy(), CategoricalAccuracy(),
               CategoricalCrossentropy(), ArgmaxMeanIoU(num_classes=6, name="mean_iou")]
    optimizer = SGD(learning_rate=initial_learning_rate, momentum=0.99, nesterov=True)
    model.compile(optimizer=optimizer, loss=weighted_categorical_crossentropy(class_weights), metrics=metrics)

    start_time = int(time.time())
    os.mkdir(f"weights/{start_time}_{model.name}/")

    metrics_to_log = ["loss", "accuracy", "categorical_accuracy", "mean_iou", "categorical_crossentropy"]
    model_path = f"weights/{start_time}_{model.name}/"
    callbacks = [
        save_model_on_epoch_end(model.name, model, model_path),
        metrics_to_csv_logger(model_path + "/batch.csv", metrics_to_log),
        CSVLogger(model_path + "/epoch.csv", separator=";"),
        LearningRateScheduler(lr_schedule(initial_lr=initial_learning_rate)),
    ]

    model.fit(training_gen, epochs=20, steps_per_epoch=steps_per_epoch,
              validation_data=validation_gen, validation_steps=validation_steps,
              callbacks=callbacks)


if __name__ == '__main__':
    # predict([218728, 165639, 115316, 27516], "weights/1595707740_unet-23D/unet-23D_epoch_20.hdf5")
    # predict([104483, 146915, 160918], "weights/1595707740_unet-23D/unet-23D_epoch_20.hdf5", mode="test")
    # exit(0)

    do_training()
