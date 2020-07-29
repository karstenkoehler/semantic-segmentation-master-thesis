import random
import os
import cv2
import numpy as np

from models.common.common import chunks, one_hot_encoding


class DataGenerator:
    def __init__(self, dataset_name, gids, batch_size, label_size, seed=0, mode="train"):
        self.rnd = random.Random(seed)
        self.gids = gids
        self.batch_size = batch_size

        self.image_base_dir = os.path.join("E:", "data", dataset_name, mode, "images")
        self.label_base_dir = os.path.join("E:", "data", dataset_name, mode, "labels")

        label = cv2.imread(os.path.join(self.label_base_dir, f"{gids[0]}.png"), cv2.IMREAD_GRAYSCALE)
        orig_label_size = label.shape[0]
        self.crop_size = label_size
        self.crop_offset = (orig_label_size - label_size) // 2
        self.crop_label = (orig_label_size != label_size)

    def get_generator(self):
        yield len(self.gids) // self.batch_size

        while True:
            self.rnd.shuffle(self.gids)
            for chunk in chunks(self.gids, self.batch_size):
                images, labels = [], []
                for gid in chunk:
                    image = cv2.imread(os.path.join(self.image_base_dir, f"{gid}.png"), cv2.IMREAD_COLOR)
                    images.append(image / 255)

                    label = cv2.imread(os.path.join(self.label_base_dir, f"{gid}.png"), cv2.IMREAD_GRAYSCALE)
                    if self.crop_label:
                        label = label[self.crop_offset:self.crop_offset + self.crop_size,
                                      self.crop_offset:self.crop_offset + self.crop_size]
                    labels.append(one_hot_encoding(label))

                yield np.array(images), np.array(labels)


def initialize_train_and_validation_generators(dataset_name, gids, batch_size, label_target_size, validation_split=0.1):
    rnd = random.Random(42)
    rnd.shuffle(gids)

    split = int(len(gids) * validation_split)
    validation = gids[:split]
    training = gids[split:]

    train_gen = DataGenerator(dataset_name, training, batch_size, label_target_size, seed=17).get_generator()
    validation_gen = DataGenerator(dataset_name, validation, batch_size, label_target_size, seed=29).get_generator()
    return train_gen, validation_gen
