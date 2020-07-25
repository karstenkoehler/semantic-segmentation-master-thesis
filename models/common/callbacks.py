import csv
import os

from tensorflow.keras.callbacks import LambdaCallback


def metrics_to_csv_logger(file_path, metrics):
    with open(file_path, "w", newline="") as file:
        wr = csv.writer(file, delimiter=";")
        wr.writerow(["batch"] + metrics)

    def callback(batch, logs):
        with open(file_path, "a", newline="") as file:
            wr = csv.writer(file, delimiter=";")
            wr.writerow([batch] + [logs[metric] for metric in metrics])

    return LambdaCallback(on_batch_end=callback)


def save_model_on_epoch_end(model_name, model, path):
    def callback(epoch, logs):
        model.save(os.path.join(path, f"{model_name}_epoch_{epoch}.hdf5"))

    return LambdaCallback(on_epoch_end=callback)
