import os
import time

import cv2
import numpy as np
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy, MeanIoU
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import set_value

from models.common.callbacks import metrics_to_csv_logger, save_model_on_epoch_end
from models.common.common import get_gids_from_database, one_hot_to_rgb, one_hot_encoding
from models.common.data_generator import initialize_train_and_validation_generators
from models.common.metrics import ArgmaxMeanIOU
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
    labels = []
    for gid in gids:
        image = cv2.imread(os.path.join("E:", "data", "unet", "train", "images", f"{gid}.png"), cv2.IMREAD_COLOR)
        images.append(image / 255)

        label = cv2.imread(os.path.join("E:", "data", "unet", "train", "labels", f"{gid}.png"), cv2.IMREAD_GRAYSCALE)
        labels.append(one_hot_encoding(label))

    pred = model.predict(np.array(images))
    losses = categorical_crossentropy(labels, pred)
    losses = np.mean(losses, axis=(1, 2))

    argmax_mean_iou = ArgmaxMeanIOU(num_classes=6)
    for idx, p in enumerate(pred):
        argmax_mean_iou.update_state(labels[idx], p)
        iou = argmax_mean_iou.result().numpy()

        print(f"{gids[idx]}: loss={losses[idx]:02f}     iou={iou:02f}")

        cv2.imwrite(f"images/{gids[idx]}-pred.png", one_hot_to_rgb(p))
        cv2.imwrite(f"images/{gids[idx]}.png", one_hot_to_rgb(labels[idx]))


def do_training():
    gids = get_gids_from_database("unet")
    training_gen, validation_gen = initialize_train_and_validation_generators("unet", gids, batch_size=4)
    steps_per_epoch = next(training_gen)
    validation_steps = next(validation_gen)

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
    # predict([218728, 165639, 115316, 27516], "weights/1595700422_unet-23D-softmax/unet-23D_epoch_0.hdf5")
    # exit(0)

    do_training()
