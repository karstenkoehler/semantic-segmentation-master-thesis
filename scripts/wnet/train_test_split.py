import psycopg2
import random
import time
import os


def add_test_set_column():
    stmt = "ALTER TABLE geom_tiles_wnet ADD test_set BOOLEAN DEFAULT FALSE;"
    with db.cursor() as cur:
        cur.execute(stmt)


def get_gids_intersecting_segment(segment_description):
    stmt = f"SELECT geom_tiles_wnet.gid FROM geom_segments, geom_tiles_wnet " \
           f"WHERE ST_Intersects(geom_segments.geom, geom_tiles_wnet.geom_label)" \
           f"AND geom_segments.segmentation_description='{segment_description}'" \
           f"ORDER BY geom_tiles_wnet.gid;"
    with db.cursor() as cur:
        cur.execute(stmt)
        return [str(row[0]) for row in cur.fetchall()]


def set_test_data(gids):
    with db.cursor() as cur:
        for chunk in chunks(gids, 50):
            gid_list = ",".join(chunk)
            stmt = f"UPDATE geom_tiles_wnet SET test_set=TRUE WHERE gid IN ({gid_list})"
            cur.execute(stmt)

    train_dir = os.path.join("E:", "data", "wnet", "train")
    test_dir = os.path.join("E:", "data", "wnet", "test")

    for gid in gids:
        try:
            os.rename(os.path.join(train_dir, "images", f"{gid}.png"), os.path.join(test_dir, "images", f"{gid}.png"))
            os.rename(os.path.join(train_dir, "labels", f"{gid}.png"), os.path.join(test_dir, "labels", f"{gid}.png"))
        except Exception:
            pass



def chunks(gids, n):
    for i in range(0, len(gids), n):
        yield gids[i:i + n]


if __name__ == '__main__':
    start = time.time()
    test_data_share = 0.1
    segments = ["water", "buildings", "agriculture", "forest", "urban greens", "traffic"]
    random.seed(42)

    db = psycopg2.connect("dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'")
    db.autocommit = True
    add_test_set_column()

    for segment in segments:
        gids = get_gids_intersecting_segment(segment)
        test_gids = random.sample(gids, int(len(gids) * test_data_share))
        set_test_data(test_gids)
        print(f"selected {len(test_gids)}/{len(gids)} {segment} tiles - {time.time() - start:.2f}")

    db.close()
