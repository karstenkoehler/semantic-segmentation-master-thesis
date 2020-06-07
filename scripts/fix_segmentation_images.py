import os
import cv2
import numpy as np
import scipy.spatial as sp
import time
import psycopg2

available_colors = [
    (240, 126, 11),  # water
    (3, 0, 208),  # buildings
    (132, 240, 235),  # agriculture
    (40, 171, 44),  # forest
    (39, 255, 154),  # urban greens
    (193, 193, 193),  # traffic
]
tree = sp.KDTree(available_colors)


def fix_image_pixels(image):
    h, w, _ = np.shape(image)
    for py in range(0, h):
        for px in range(0, w):
            input_color = tuple(image[py][px].tolist())
            if input_color in available_colors:
                continue

            if px > 0:
                image[py][px] = image[py][px - 1]
            elif py > 0:
                image[py][px] = image[py - 1][px]
            else:
                _, closest_index = tree.query(input_color)
                image[py][px] = available_colors[closest_index]
    return image


def fix_and_save_image(gid, base_dir=os.path.join("E:", "data", "segmentation_labels_fixed")):
    file_path = os.path.join(base_dir, f"{gid}.png")
    image = cv2.imread(file_path)
    image = fix_image_pixels(image)
    cv2.imwrite(file_path, image)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def select_tile_gids_with_multiple_intersections(low, high):
    return """SELECT geom_tiles.gid FROM geom_tiles, geom_segments
    WHERE geom_tiles.gid BETWEEN {0} AND {1} AND ST_Intersects(geom_tiles.geom, geom_segments.geom)
    GROUP BY geom_tiles.gid HAVING COUNT(geom_tiles.gid) > 1;""".format(low, high)


def get_gids_to_fix(chunk_size=3000):
    print("fetching all tile GIDs that need to be fixed")
    db = psycopg2.connect("dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'")
    gids_to_fix = []
    with db.cursor() as cur:
        cur.execute("SELECT gid FROM geom_tiles ORDER BY gid")
        gids = [int(gid[0]) for gid in cur.fetchall()]
        gid_count = len(gids)

        for chunk in chunks(gids, chunk_size):
            low, high = min(chunk), max(chunk)
            cur.execute(select_tile_gids_with_multiple_intersections(low, high))
            gids_to_fix += [int(gid[0]) for gid in cur.fetchall()]
            print(f"processed {high}/{gid_count} tiles")

    print(f"found {len(gids_to_fix)} tiles that need fixing")
    db.close()
    return gids_to_fix


if __name__ == '__main__':
    start = time.time()

    gids = get_gids_to_fix()
    fixed = 0
    for chunk in chunks(gids, 100):
        for gid in chunk:
            fix_and_save_image(gid)
            fixed += 1
        print(f"fixed {fixed}/{len(gids)} images")

    print(time.time() - start)
