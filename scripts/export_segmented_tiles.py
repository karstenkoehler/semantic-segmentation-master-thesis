import psycopg2
import os


def get_polygons():
    db = psycopg2.connect("dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'")
    with db.cursor() as cur:
        cur.execute("SELECT gid, ST_AsText(geom) FROM geom_tiles")
        value = cur.fetchone()
        while value:
            yield value
            value = cur.fetchone()
    db.close()


def add_layer():
    uri = QgsDataSourceUri()
    # set host name, port, database name, username and password
    uri.setConnection("localhost", "5432", "dop10rgbi_nrw", "postgres", "root")
    # set database schema, table name, geometry column and optionally
    # subset (WHERE clause)
    uri.setDataSource("public", "geom_bounds", "geom")

    vlayer = QgsVectorLayer(uri.uri(False), "testlayer", "postgres")


def get_layers():
    layer_names = ["1_water", "2_buildings", "3_agriculture", "4_forest", "5_urban_greens", "6_traffic"]
    return [QgsProject.instance().mapLayersByName(name)[0] for name in layer_names]


def get_map_settings(layers, geom_string, img_size=250):
    settings = QgsMapSettings()
    settings.setLayers(layers)
    settings.setBackgroundColor(QColor(0, 0, 0))
    settings.setOutputSize(QSize(img_size, img_size))

    geom = QgsGeometry.fromWkt(geom_string)
    settings.setExtent(geom.boundingBox())

    return settings


out_folder = os.path.join("E:", "data", "segmentation_labels")
layers = get_layers()

for gid, polygon in get_polygons():
    settings = get_map_settings(layers, polygon)
    render = QgsMapRendererSequentialJob(settings)
    image_path = os.path.join(out_folder, f"{gid}.png")

    render.start()
    render.waitForFinished()
    img = render.renderedImage()
    img.save(image_path, "png")
