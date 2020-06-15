import psycopg2
import os
import time


def get_polygons(table_suffix=""):
    db = psycopg2.connect("dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'")
    with db.cursor() as cur:
        cur.execute(f"SELECT gid, ST_AsText(geom_image), ST_AsText(geom_label) FROM geom_tiles_{table_suffix}")
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


def get_segmentation_layer():
    layer_names = ["1_water", "2_buildings", "3_agriculture", "4_forest", "5_urban_greens", "6_traffic"]
    return [QgsProject.instance().mapLayersByName(name)[0] for name in layer_names]


def get_dop_rgb_layer():
    return [QgsProject.instance().mapLayersByName(name)[0] for name in ["dop_rgb"]]


def get_map_settings(layers, geom_string, img_size=250):
    settings = QgsMapSettings()
    settings.setLayers(layers)
    settings.setBackgroundColor(QColor(0, 0, 0))
    settings.setOutputSize(QSize(img_size, img_size))

    geom = QgsGeometry.fromWkt(geom_string)
    settings.setExtent(geom.boundingBox())

    return settings


def render_and_save_image(bounding_box, layers, image_size, save_path):
    settings = get_map_settings(layers, bounding_box, image_size)
    render = QgsMapRendererSequentialJob(settings)

    render.start()
    render.waitForFinished()
    img = render.renderedImage()
    img.save(save_path, "png")


base_dir = os.path.join("E:", "data", "unet")
segmentation_layer = get_segmentation_layer()
dop_layer = get_dop_rgb_layer()

count = 0
start = time.time()
for gid, image_box, label_box in get_polygons("unet"):
    image_path = os.path.join(base_dir, "images", f"{gid}.png")
    label_path = os.path.join(base_dir, "labels", f"{gid}.png")

    render_and_save_image(image_box, dop_layer, 572, image_path)
    render_and_save_image(label_box, segmentation_layer, 388, label_path)
    if count % 100:
        print(f"{count} - {time.time() - start}s", flush=True)
