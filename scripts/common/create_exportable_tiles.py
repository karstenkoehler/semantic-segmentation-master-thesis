import random

import psycopg2
import time

from models.common.common import chunks
from scripts.common.constants import SEGMENTATION_CATEGORIES


def create_table(db, table_suffix="", srid=25832):
    with db.cursor() as cur:
        stmt = f"CREATE TABLE public.geom_tiles_{table_suffix} (gid SERIAL PRIMARY KEY);"
        print(stmt)
        cur.execute(stmt)
        stmt = f"SELECT AddGeometryColumn('public', 'geom_tiles_{table_suffix}', 'geom_image', {srid}, 'POLYGON', 2)"
        print(stmt)
        cur.execute(stmt)
        stmt = f"SELECT AddGeometryColumn('public', 'geom_tiles_{table_suffix}', 'geom_label', {srid}, 'POLYGON', 2)"
        print(stmt)
        cur.execute(stmt)


def get_min_max(db):
    with db.cursor() as cur:
        stmt = "SELECT ST_XMin(geom), ST_XMax(geom), ST_YMin(geom), ST_YMax(geom) FROM geom_bounds"
        print(stmt)
        cur.execute(stmt)
        result = cur.fetchone()
        return (result[0], result[1]), (result[2], result[3])


def yield_tile_coordinates(boundaries, tile_size, label_size=None, label_offset=0.0):
    (min_x, max_x), (min_y, max_y) = boundaries
    if label_size is None:
        label_size = tile_size

    x_steps = int((max_x - min_x) // label_size)
    y_steps = int((max_y - min_y) // label_size)

    for x_step in range(x_steps):
        for y_step in range(y_steps):
            x = min_x + (x_step * label_size)
            y = min_y + (y_step * label_size)

            image = create_envelope_string(x, x + tile_size, y, y + tile_size)
            label = create_envelope_string(x + label_offset, x + label_offset + label_size, y + label_offset,
                                           y + label_offset + label_size)
            yield image, label


def create_envelope_string(min_x, max_x, min_y, max_y, srid=25832):
    return f"ST_MakeEnvelope({min_x}, {min_y},{max_x}, {max_y}, {srid})"


def generate_tiles(db, generator, table_suffix=""):
    print("generating all tiles...")
    for image_geom, label_geom in generator:
        stmt = f"INSERT INTO geom_tiles_{table_suffix} (geom_image, geom_label) VALUES ({image_geom}, {label_geom})"
        with db.cursor() as cur:
            cur.execute(stmt)


def delete_outer_tiles(db, table_suffix=""):
    stmt = f"DELETE FROM geom_tiles_{table_suffix} WHERE " \
           f"NOT (SELECT ST_Contains(geom_bounds.geom, geom_tiles_{table_suffix}.geom_image) FROM geom_bounds)"
    with db.cursor() as cur:
        print(stmt)
        cur.execute(stmt)


def add_multisegment_column(db, table_suffix=""):
    print("adding multisegment column...")
    alter_stmt = f"ALTER TABLE geom_tiles_{table_suffix} ADD segment_count INTEGER DEFAULT 0;"
    update_stmt = f"UPDATE geom_tiles_{table_suffix} " \
                  f"SET segment_count=subquery.count " \
                  f"FROM (SELECT geom_tiles_{table_suffix}.gid, COUNT(geom_tiles_{table_suffix}.gid) count " \
                  f"  FROM geom_tiles_{table_suffix}, geom_segments " \
                  f"  WHERE ST_Intersects(geom_tiles_{table_suffix}.geom_label, geom_segments.geom) " \
                  f"  GROUP BY geom_tiles_{table_suffix}.gid) AS subquery " \
                  f"WHERE geom_tiles_{table_suffix}.gid = subquery.gid;"

    with db.cursor() as cur:
        print(alter_stmt)
        cur.execute(alter_stmt)
        print(update_stmt)
        cur.execute(update_stmt)


def add_testset_column(db, table_suffix="", test_data_split=0.1):
    with db.cursor() as cur:
        alter_stmt = f"ALTER TABLE geom_tiles_{table_suffix} ADD test_set BOOLEAN DEFAULT FALSE;"
        print(alter_stmt)
        cur.execute(alter_stmt)

    random.seed(1654812)
    for segment in SEGMENTATION_CATEGORIES:
        gids = get_gids_intersecting_segment(db, segment, table_suffix=table_suffix)
        test_gids = random.sample(gids, int(len(gids) * test_data_split))
        with db.cursor() as cur:
            for chunk in chunks(test_gids, 50):
                gid_list = ",".join(chunk)
                alter_stmt = f"UPDATE geom_tiles_{table_suffix} SET test_set=TRUE WHERE gid IN ({gid_list})"
                cur.execute(alter_stmt)


def get_gids_intersecting_segment(db, segment_description, table_suffix=""):
    stmt = f"SELECT geom_tiles_{table_suffix}.gid FROM geom_segments, geom_tiles_{table_suffix} " \
           f"WHERE ST_Intersects(geom_segments.geom, geom_tiles_{table_suffix}.geom_label)" \
           f"AND geom_segments.segmentation_description='{segment_description}'" \
           f"ORDER BY geom_tiles_{table_suffix}.gid;"
    with db.cursor() as cur:
        cur.execute(stmt)
        return [str(row[0]) for row in cur.fetchall()]


def create_exportable_tile_table(table_suffix, tile_size, label_size):
    db = psycopg2.connect("dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'")
    db.autocommit = True

    create_table(db, table_suffix=table_suffix)
    bounds = get_min_max(db, )

    start = time.time()
    offset = (tile_size - label_size) / 2
    gen = yield_tile_coordinates(bounds, tile_size=tile_size, label_size=label_size, label_offset=offset)
    generate_tiles(db, gen, table_suffix=table_suffix)
    delete_outer_tiles(db, table_suffix=table_suffix)
    add_multisegment_column(db, table_suffix=table_suffix)
    add_testset_column(db, table_suffix=table_suffix)
    print(f"done - {time.time() - start}s")

    db.close()
