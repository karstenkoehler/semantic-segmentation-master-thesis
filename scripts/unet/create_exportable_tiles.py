import psycopg2
import time


def create_table(table_suffix="", srid=25832):
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


def get_min_max():
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


def generate_tiles(generator, table_suffix=""):
    print("generating all tiles...")
    for image_geom, label_geom in generator:
        stmt = f"INSERT INTO geom_tiles_{table_suffix} (geom_image, geom_label) VALUES ({image_geom}, {label_geom})"
        with db.cursor() as cur:
            cur.execute(stmt)


def delete_outer_tiles(table_suffix=""):
    stmt = f"DELETE FROM geom_tiles_{table_suffix} WHERE " \
           f"NOT (SELECT ST_Contains(geom_bounds.geom, geom_tiles_{table_suffix}.geom_image) FROM geom_bounds)"
    with db.cursor() as cur:
        print(stmt)
        cur.execute(stmt)


if __name__ == '__main__':
    db = psycopg2.connect("dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'")
    db.autocommit = True

    table_suffix = "unet"
    create_table(table_suffix=table_suffix)
    bounds = get_min_max()

    start = time.time()
    gen = yield_tile_coordinates(bounds, 57.2, label_size=38.8, label_offset=9.2)
    generate_tiles(gen, table_suffix=table_suffix)
    delete_outer_tiles(table_suffix=table_suffix)
    print(f"done - {time.time() - start}s")

    db.close()
