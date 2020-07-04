import os
import cv2
import numpy as np
import scipy.spatial as sp
import time

base_dir = os.path.join("E:", "data", "densenet", "labels")
valid_colors = [
    (240, 126, 11),  # water
    (3, 0, 208),  # buildings
    (132, 240, 235),  # agriculture
    (40, 171, 44),  # forest
    (39, 255, 154),  # urban greens
    (193, 193, 193),  # traffic
]
tree = sp.KDTree(valid_colors)


def validate_image(image):
    # quickly check if the whole image belongs to one valid segment
    if tuple(image[0][0]) in valid_colors and np.all(image == image[0][0]):
        return True

    # check if each pixel is assigned to a valid category
    segments = []
    for col in valid_colors:
        segments += [image == col]

    return np.all(np.logical_or.reduce(segments))


def fix_image_pixels(image):
    h, w, _ = np.shape(image)
    for py in range(0, h):
        for px in range(0, w):
            input_color = tuple(image[py][px].tolist())
            if input_color in valid_colors:
                continue

            if px > 0:
                image[py][px] = image[py][px - 1]
            elif py > 0:
                image[py][px] = image[py - 1][px]
            else:
                _, closest_index = tree.query(input_color)
                image[py][px] = valid_colors[closest_index]
    return image


def fix_and_save_image(file_name):
    file_path = os.path.join(base_dir, file_name)
    image = cv2.imread(file_path)
    image = fix_image_pixels(image)
    cv2.imwrite(file_path, image)


if __name__ == '__main__':
    start = time.time()

    for _, _, files in os.walk(base_dir):
        print(f"{len(files)} files to check")
        invalid_files = set()

        count = 0
        for f in files:
            count += 1
            image = cv2.imread(os.path.join(base_dir, f))
            if not validate_image(image):
                invalid_files.add(f)

            if count % 100 == 0:
                print(f"{count}/{len(files)} - ({time.time() - start:.2f})")

        print(f"found {len(invalid_files)} invalid files:")
        for file_name in invalid_files:
            fix_and_save_image(file_name)
            print(f"fixed {file_name}")
