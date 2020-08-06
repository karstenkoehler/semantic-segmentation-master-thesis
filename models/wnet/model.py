import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.metrics import Accuracy, CategoricalAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
from tensorflow.keras.losses import MSE
from tensorflow.keras.metrics import CategoricalCrossentropy

from models.common.callbacks import metrics_to_csv_logger, save_model_on_epoch_end
from models.common.common import get_gids_from_database, split_to_tiles
from models.common.data_generator import initialize_train_and_validation_generators
from models.wnet.wnet import WNet


def predict(gids, model_path, num_classes=1000):
    model = load_model(model_path)

    for gid in gids:
        image = cv2.imread(os.path.join("E:", "data", "wnet", "images", f"{gid}.png"), cv2.IMREAD_COLOR)
        images = split_to_tiles(image / 255, 224)

        final_prediction = np.empty([0, 2240, 1])
        row = np.empty([224, 0, 1])

        for idx, img in enumerate(images):
            pred = model.predict(np.array([images[idx]]))
            tile = np.argmax(pred[0], axis=2).reshape((224, 224, 1))
            # print(np.unique(tile.reshape((224*224,))))
            row = np.hstack([row, tile])
            if row.shape[1] >= 2240:
                # row = np.argmax(row, axis=2)
                final_prediction = np.vstack([final_prediction, row])
                row = np.empty([224, 0, 1])

        # FIXME: use common.one_hot_to_rgb instead
        cmap = plt.cm.get_cmap("hsv", num_classes)
        final_prediction = final_prediction.reshape((2240, 2240))
        plt.imsave(f"images/{gid}-pred.png", final_prediction, cmap=cmap)


def restore(gids, model_path):
    model = load_model(model_path)

    for gid in gids:
        image = cv2.imread(os.path.join("E:", "data", "wnet", "images", f"{gid}.png"), cv2.IMREAD_COLOR)
        images = split_to_tiles(image / 255, 224)

        final_prediction = np.empty([0, 2240, 3])
        row = np.empty([224, 0, 3])

        for idx, img in enumerate(images):
            pred = model.predict(np.array([images[idx]]))

            row = np.hstack([row, pred[0]])
            if row.shape[1] >= 2240:
                final_prediction = np.vstack([final_prediction, row])
                row = np.empty([224, 0, 3])

        final_prediction = final_prediction * 255
        # print(f"restore min_max: {np.min(final_prediction)}, {np.max(final_prediction)}")
        cv2.imwrite(f"images/{gid}-restore.png", final_prediction)


def do_training(initial_learning_rate=0.001):
    gids = get_gids_from_database("wnet")
    training_gen, validation_gen = initialize_train_and_validation_generators("wnet", gids, batch_size=10,
                                                                              label_target_size=256, use_image_as_label=True)
    steps_per_epoch = next(training_gen)
    validation_steps = next(validation_gen)

    full_model, encoder_model = WNet()
    metrics = [Accuracy(), CategoricalAccuracy(), CategoricalCrossentropy()]
    optimizer = Adam(lr=initial_learning_rate)
    full_model.compile(optimizer=optimizer, loss=MSE, metrics=metrics)

    start_time = int(time.time())
    model_path = f"weights/{start_time}_{full_model.name}/"
    os.mkdir(model_path)

    metrics_to_log = [metric.name for metric in metrics]
    callbacks = [
        save_model_on_epoch_end(full_model.name, full_model, model_path),
        save_model_on_epoch_end(encoder_model.name, encoder_model, model_path),
        metrics_to_csv_logger(model_path + "/batch.csv", ["loss"] + metrics_to_log),
        CSVLogger(model_path + "/epoch.csv", separator=";"),
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4, min_lr=1e-6),
    ]
    full_model.fit(training_gen, epochs=50, steps_per_epoch=steps_per_epoch,
                   validation_data=validation_gen, validation_steps=validation_steps,
                   callbacks=callbacks)


if __name__ == '__main__':
    # predict([51], "weights/1595520768/epoch_0_encoder_model.hdf5")
    # restore([51], "weights/1595520768/run-00__epoch-05__val-loss-1.48.hdf5")
    # exit(0)

    do_training()
