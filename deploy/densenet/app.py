import os
import glob
import math
import cv2
import numpy as np

from tensorflow.keras.models import load_model

PREDICTION_TILE_SIZE = 224

# color values for the segmentation categories
LABEL_RGB_VALUES = [
    (3, 0, 208),  # buildings
    (240, 126, 11),  # water
    (40, 171, 44),  # forest
    (193, 193, 193),  # traffic
    (39, 255, 154),  # urban greens
    (132, 240, 235),  # agriculture
]


def one_hot_to_rgb(prediction, color_palette=None):
    if np.ndim(prediction) != 3:
        raise ValueError("prediction should have 3 dimensions")

    if color_palette is None:
        color_palette = np.array(LABEL_RGB_VALUES)

    classes = np.argmax(prediction, axis=2)

    rgb_encoded = np.zeros(classes.shape[:2] + (3,))
    for idx, col in enumerate(color_palette):
        rgb_encoded[classes == idx] = col
    return rgb_encoded


def map_to_color(color_palette):
    def fn(x):
        result = color_palette(x)
        return result[:, :-1]

    return fn


def split_to_tiles(img, tile_size):
    tiles = []
    x_steps = img.shape[0] // tile_size
    y_steps = img.shape[1] // tile_size

    for x in range(x_steps):
        for y in range(y_steps):
            tile = img[x * tile_size:(x + 1) * tile_size, y * tile_size:(y + 1) * tile_size, :]
            tiles.append(tile)

    return tiles


def merge_tiles(tiles):
    tiles_per_row = int(math.sqrt(len(tiles)))
    image = np.zeros((tiles_per_row * PREDICTION_TILE_SIZE, 0, 6))

    for x in range(tiles_per_row):
        row = np.zeros((0, 224, 6))
        for y in range(tiles_per_row):
            row = np.concatenate([row, tiles[y * tiles_per_row + x]], axis=0)
        image = np.concatenate([image, row], axis=1)
    return image


def predict(model, image):
    image = image / 255
    tiles = split_to_tiles(image, 224)
    pred = model.predict(np.array(tiles))
    merged = merge_tiles(pred.tolist())
    return one_hot_to_rgb(merged)


if __name__ == '__main__':
    model = load_model("/thesis/model/model.hdf5", compile=False)

    for img_path in glob.glob("/images/*.png"):
        print(f"predicting {img_path} ...")
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image.shape[0] % PREDICTION_TILE_SIZE != 0 or image.shape[1] % PREDICTION_TILE_SIZE != 0:
            print(f"WARN: {img_path} image dimensions must be multiple of {PREDICTION_TILE_SIZE}")

        predicted_path = os.path.join("/predictions", os.path.basename(img_path))
        cv2.imwrite(predicted_path, predict(model, image))
