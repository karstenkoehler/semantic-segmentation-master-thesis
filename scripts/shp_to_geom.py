import os
import subprocess
import psycopg2


def import_shapefiles(files):
    # create intermediate shape tables
    for f in files:
        cmd = f"shp2pgsql -s 25832 -g geom -e -S -t 2D -N skip {os.path.join(base_path, f)}.shp public.geom_tmp_{f} | psql -d dop10rgbi_nrw -q"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)


def create_tmp_segmentation_table():
    # create temporary segmentation table
    with db.cursor() as cur:
        stmt = """CREATE TABLE public.geom_tmp (
          gid SERIAL PRIMARY KEY, 
          object_type INTEGER, 
          object_description TEXT, 
          segmentation_category INTEGER, 
          geom geometry);"""
        cur.execute(stmt)


def filter_and_cut_shapes(files):
    # filter and cut all geoms, import to segmentation table
    base_statement = """INSERT INTO public.geom_tmp (object_type, object_description, geom)
        (SELECT geom_tmp_{0}.objart::int, geom_tmp_{0}.objart_txt, ST_Intersection(geom_tmp_{0}.geom, geom_bounds.geom)
        FROM geom_tmp_{0}, geom_bounds
        WHERE ST_Intersects(geom_tmp_{0}.geom, geom_bounds.geom));
    """
    with db.cursor() as cur:
        for f in files:
            stmt = base_statement.format(f)
            print(stmt)
            cur.execute(stmt)


def create_final_segmentation_table():
    # create final segmentation table
    with db.cursor() as cur:
        stmt = """CREATE TABLE public.geom_segments (
          gid SERIAL PRIMARY KEY, 
          segmentation_description TEXT,
          geom geometry);"""
        cur.execute(stmt)


def map_and_merge_segments():
    # map object type to segmentation category
    object_type_mapping = {
        "water": [44001, 44005, 44006],
        "buildings": [41002, 41005, 41007, 41010],
        "agriculture": [43001, 43007],
        "forest": [43002, 43003, 43004],
        "urban greens": [41008, 41009],
        "traffic": [42001, 42009, 42010],
    }

    with db.cursor() as cur:
        for cat, ids in object_type_mapping.items():
            ids = [str(x) for x in ids]
            stmt = """INSERT INTO geom_segments (segmentation_description, geom) 
                SELECT '{0}' segmentation_category, ST_Union(geom) 
                FROM geom_tmp
                WHERE object_type IN ({1});""".format(cat, ','.join(ids))
            print(stmt)
            cur.execute(stmt)


def clean_up_database(files):
    # remove intermediate shape tables
    with db.cursor() as cur:
        for f in files:
            stmt = f"DROP TABLE public.geom_tmp_{f};"
            print(stmt)
            cur.execute(stmt)

        stmt = "DROP TABLE public.geom_tmp"
        print(stmt)
        cur.execute(stmt)


if __name__ == '__main__':
    base_path = os.path.join("E:", "data", "dlm_segmentation_shapes")
    files = ["ver01_f", "gew01_f", "ver03_f", "veg03_f", "veg01_f", "veg02_f", "sie02_f"]

    db = psycopg2.connect("dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'")
    db.autocommit = True

    import_shapefiles(files)
    create_tmp_segmentation_table()
    filter_and_cut_shapes(files)
    create_final_segmentation_table()
    map_and_merge_segments()
    clean_up_database(files)

    db.close()
