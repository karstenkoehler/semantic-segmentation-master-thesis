import os
import glob
import math
import cv2
import numpy as np

from tensorflow.keras.models import load_model

INPUT_TILE_SIZE = 572
PREDICTION_TILE_SIZE = 388

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


def split_to_tiles(image):
    tiles = []
    x_steps = image.shape[0] // PREDICTION_TILE_SIZE
    y_steps = image.shape[1] // PREDICTION_TILE_SIZE

    offset = (INPUT_TILE_SIZE - PREDICTION_TILE_SIZE)
    extended_image = add_mirrored_edges(image, offset // 2)

    for x in range(x_steps):
        for y in range(y_steps):
            tile = extended_image[x * PREDICTION_TILE_SIZE:(x + 1) * PREDICTION_TILE_SIZE + offset,
                   y * PREDICTION_TILE_SIZE:(y + 1) * PREDICTION_TILE_SIZE + offset, :]
            tiles.append(tile)

    return tiles


def merge_tiles(tiles):
    tiles_per_row = int(math.sqrt(len(tiles)))
    image = np.zeros((tiles_per_row * PREDICTION_TILE_SIZE, 0, 6))

    for x in range(tiles_per_row):
        row = np.zeros((0, PREDICTION_TILE_SIZE, 6))
        for y in range(tiles_per_row):
            row = np.concatenate([row, tiles[y * tiles_per_row + x]], axis=0)
        image = np.concatenate([image, row], axis=1)
    return image


def add_mirrored_edges(image, context=100):
    height = image.shape[0]
    width = image.shape[1]

    top_edge = image[0:context, :, :]
    top_edge = cv2.flip(top_edge, 0)

    bottom_edge = image[height - context:height, :, :]
    bottom_edge = cv2.flip(bottom_edge, 0)

    image = np.concatenate([top_edge, image, bottom_edge], axis=0)

    left_edge = image[:, 0:context, :]
    left_edge = cv2.flip(left_edge, 1)

    right_edge = image[:, width - context:width, :]
    right_edge = cv2.flip(right_edge, 1)

    image = np.concatenate([left_edge, image, right_edge], axis=1)
    return image


def predict(model, image):
    image = image / 255
    tiles = split_to_tiles(image)
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
