import os
import subprocess
import psycopg2


def import_shapefiles(files):
    # create intermediate shape tables
    for f in files:
        cmd = f"shp2pgsql -s 25832 -g geom -e -S -t 2D -N skip {os.path.join(base_path, f)}.shp public.geom_segments_{f} | psql -d dop10rgbi_nrw -q"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)


def create_segmentation_table():
    # create final segmentation table
    cur = db.cursor()
    stmt = """CREATE TABLE public.geom_segments (
      gid SERIAL PRIMARY KEY, 
      object_type INTEGER, 
      object_description TEXT, 
      segmentation_category INTEGER, 
      geom geometry);"""
    cur.execute(stmt)
    cur.close()


def filter_and_cut_shapes(files):
    # filter and cut all geoms, import to segmentation table
    base_statement = """INSERT INTO public.geom_segments (object_type, object_description, geom)
        (SELECT geom_segments_{0}.objart::int, geom_segments_{0}.objart_txt, ST_Intersection(geom_segments_{0}.geom, geom_bounds.geom)
        FROM geom_segments_{0}, geom_bounds
        WHERE ST_Intersects(geom_segments_{0}.geom, geom_bounds.geom));
    """
    cur = db.cursor()
    for f in files:
        stmt = base_statement.format(f)
        print(stmt)
        cur.execute(stmt)
    cur.close()


def map_to_segments():
    # map object type to segmentation category
    object_type_mapping = {
        # 0: water, 1: buildings, 2: agriculture, 3: forest, 4: urban greens, 5: traffic
        44001: 0, 44005: 0, 44006: 0, 41007: 1, 41009: 4, 41002: 1,
        41010: 1, 41008: 4, 41005: 1, 43001: 2, 43002: 3, 43003: 3,
        43004: 3, 43007: 2, 42009: 5, 42001: 5, 42010: 5
    }

    cur = db.cursor()
    for object_type, object_segment in object_type_mapping.items():
        stmt = f"UPDATE public.geom_segments SET segmentation_category={object_segment} WHERE object_type={object_type}"
        print(stmt)
        cur.execute(stmt)
    cur.close()


def clean_up_database(files):
    # remove intermediate shape tables
    cur = db.cursor()
    for f in files:
        stmt = f"DROP TABLE public.geom_segments_{f};"
        print(stmt)
        cur.execute(stmt)
    cur.close()


if __name__ == '__main__':
    base_path = os.path.join("E:", "data", "raw_dlm")
    files = ["ver01_f", "gew01_f", "ver03_f", "veg03_f", "veg01_f", "veg02_f", "sie02_f"]

    db = psycopg2.connect("dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'")
    db.autocommit = True

    import_shapefiles(files)
    create_segmentation_table()
    filter_and_cut_shapes(files)
    map_to_segments()
    clean_up_database(files)

    db.close()
