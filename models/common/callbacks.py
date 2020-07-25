import csv
import os

from keras.callbacks import LambdaCallback


def metrics_to_csv_logger(file_path):
    with open(file_path, "w", newline="") as file:
        wr = csv.writer(file, delimiter=";")
        wr.writerow(["batch", "loss", "accuracy", "categorical_accuracy"])

    def callback(batch, logs):
        with open(file_path, "a", newline="") as file:
            wr = csv.writer(file, delimiter=";")
            wr.writerow([batch, logs["loss"], logs["accuracy"], logs["categorical_accuracy"]])

    return LambdaCallback(on_batch_end=callback)


def save_model_on_epoch_end(model_name, model, path):
    def callback(epoch, logs):
        model.save(os.path.join(path, f"{model_name}_epoch_{epoch}.hdf5"))

    return LambdaCallback(on_epoch_end=callback)
