import os
import subprocess
import time
import psycopg2


def raster_to_pgsql(file_path, bands, table_name, srid=25832, database="dop10rgbi_nrw"):
    # -t [tile size] -b [raster bands] -s [SRID]]
    # -F  add column with filename
    # -e  don't use transactions
    # -a  insert into table
    cmd = f"raster2pgsql -t 1000x1000 -b {bands} -F -I -s {srid} -e -a {file_path} {table_name} | psql -d {database} -q"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)


def create_table(table_name):
    stmt = f"CREATE TABLE public.{table_name} (rid serial PRIMARY KEY, rast raster, filename text);"
    print(stmt)
    with db.cursor() as cur:
        cur.execute(stmt)


def create_index(table_name, raster_column="rast"):
    stmt = f"CREATE INDEX ON public.{table_name} USING gist (st_convexhull({raster_column}));"
    print(stmt)
    with db.cursor() as cur:
        cur.execute(stmt)


def add_constraints(table_name, raster_column="rast"):
    stmt = f"SELECT AddRasterConstraints('public','{table_name}','{raster_column}'" \
           f",TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,FALSE,TRUE,TRUE,TRUE,TRUE,TRUE);"
    print(stmt)
    with db.cursor() as cur:
        cur.execute(stmt)


def create_boundary_table():
    stmt = "CREATE TABLE geom_bounds AS (SELECT 1 gid, ST_Union(ST_Envelope(rast)) geom FROM dop_nir);"
    print(stmt)
    with db.cursor() as cur:
        cur.execute(stmt)


if __name__ == '__main__':
    base_dir = os.path.join("E:", "data", "dop_nir_tif")
    start = time.time()

    db = psycopg2.connect("dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'")
    db.autocommit = True

    create_table("dop_rgb")
    create_table("dop_nir")
    print(f"created tables - {time.time() - start:.2f}")

    count = 0
    for _, _, files in os.walk(base_dir):
        for f in files:
            file_path = os.path.join(base_dir, f)
            raster_to_pgsql(file_path, "1-3", "dop_rgb")
            raster_to_pgsql(file_path, "4", "dop_nir")
            count += 1
            print(f"{count}/{len(files)} - {time.time() - start:.2f}")

    create_index("dop_rgb")
    create_index("dop_nir")
    print(f"created indices - {time.time() - start:.2f}")

    add_constraints("dop_rgb")
    add_constraints("dop_nir")
    print(f"added constraints - {time.time() - start:.2f}")

    create_boundary_table()
    print(f"added constraints - {time.time() - start:.2f}")

    db.close()
