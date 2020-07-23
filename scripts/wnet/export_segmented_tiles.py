import psycopg2
import os
import time
import queue


class TileRenderWorker(QgsTask):
    def __init__(self, worker_id, image_size, label_size):
        super().__init__()
        self.worker_id = worker_id
        self.image_size = image_size
        self.label_size = label_size

    def run(self):
        # TODO: adjust base dir
        base_dir = os.path.join("E:", "data", "wnet")
        segmentation_layer = get_segmentation_layer()
        dop_layer = get_dop_rgb_layer()

        while True:
            try:
                (gid, image_box, label_box) = global_queue.get(block=True, timeout=5)

                image_path = os.path.join(base_dir, "images", f"{gid}.png")
                label_path = os.path.join(base_dir, "labels", f"{gid}.png")

                render_and_save_image(image_box, dop_layer, self.image_size, image_path)
                render_and_save_image(label_box, segmentation_layer, self.label_size, label_path)
                global_queue.task_done()

            except queue.Empty:
                return True
            except Exception as e:
                QgsMessageLog.logMessage(f"{e}")
                pass

    def finished(self, result):
        QgsMessageLog.logMessage(f"worker {self.worker_id} finished")


def add_tiles_to_global_queue(table_suffix=""):
    with db.cursor() as cur:
        cur.execute(f"SELECT gid, ST_AsText(geom_image), ST_AsText(geom_label) FROM geom_tiles_{table_suffix}")

        while True:
            value = cur.fetchone()
            if value is None:
                break

            global_queue.put(tuple(value))


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


global_queue = queue.Queue(maxsize=150)
db = psycopg2.connect("dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'")

for i in range(5):
    worker = TileRenderWorker(i, 2240, 2240)
    QgsApplication.taskManager().addTask(worker)

start = time.time()
add_tiles_to_global_queue(table_suffix="wnet")
global_queue.join()
db.close()
QgsMessageLog.logMessage(f"done - {time.time() - start:.2f}")
