import os

import cv2
import numpy as np
import scipy.spatial as sp

from scripts.common.constants import LABEL_RGB_VALUES


def validate_image(image):
    segments = []
    for col in LABEL_RGB_VALUES:
        segments += [image == col]

    return np.all(np.logical_or.reduce(segments))


def fix_image_pixels(image, tree):
    h, w, _ = np.shape(image)
    for py in range(0, h):
        for px in range(0, w):
            input_color = tuple(image[py][px].tolist())
            if input_color in LABEL_RGB_VALUES:
                continue

            if px > 0:
                image[py][px] = image[py][px - 1]
            elif py > 0:
                image[py][px] = image[py - 1][px]
            else:
                _, closest_index = tree.query(input_color)
                image[py][px] = LABEL_RGB_VALUES[closest_index]
    return image


def fix_labels_in_directory(base_dir, tree):
    _, _, files = next(os.walk(base_dir))
    print(f"{len(files)} files to check in {base_dir}")
    count, fixed = 0, 0

    for file_name in files:
        file_path = os.path.join(base_dir, file_name)
        image = cv2.imread(file_path)
        if not validate_image(image):
            image = fix_image_pixels(image, tree)
            cv2.imwrite(file_path, image)
            fixed += 1

        count += 1
        if count % 100 == 0:
            print(f"{count}/{len(files)}  ({fixed} fixed so far)")

    print(f"done - fixed {fixed} files in total")


def fix_label_interpolation(dataset_name=""):
    train_base_dir = os.path.join("E:", "data", dataset_name, "train", "labels")
    test_base_dir = os.path.join("E:", "data", dataset_name, "test", "labels")

    tree = sp.KDTree(LABEL_RGB_VALUES)

    fix_labels_in_directory(train_base_dir, tree)
    fix_labels_in_directory(test_base_dir, tree)
