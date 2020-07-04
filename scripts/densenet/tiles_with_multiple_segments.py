import psycopg2

db = psycopg2.connect("dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'")

with db.cursor() as cur:
    stmt = """SELECT geom_tiles_densenet.gid FROM geom_tiles_densenet, geom_segments
        WHERE ST_Intersects(geom_tiles_densenet.geom_label, geom_segments.geom)
        GROUP BY geom_tiles_densenet.gid HAVING COUNT(geom_tiles_densenet.gid) > 1;"""
    cur.execute(stmt)
    gids = [str(gid[0]) for gid in cur.fetchall()]

    with open("../../models/densenet/gids_with_multiple_segments.txt", 'w') as f:
        f.write("\n".join(gids))

db.close()