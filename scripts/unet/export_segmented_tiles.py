import psycopg2
import os
import time
import queue


class TileRenderWorker(QgsTask):
    def __init__(self, worker_id, dataset_name, image_size, label_size):
        super().__init__()
        self.worker_id = worker_id
        self.image_size = image_size
        self.label_size = label_size

        self.train_base_dir = os.path.join("E:", "data", dataset_name, "train")
        self.test_base_dir = os.path.join("E:", "data", dataset_name, "test")

        os.makedirs(os.path.join(self.train_base_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.train_base_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(self.test_base_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.test_base_dir, "labels"), exist_ok=True)

        self.segmentation_layer = self.get_segmentation_layer()
        self.dop_layer = self.get_dop_rgb_layer()

    def get_segmentation_layer(self):
        layer_names = ["1_water", "2_buildings", "3_agriculture", "4_forest", "5_urban_greens", "6_traffic"]
        return [QgsProject.instance().mapLayersByName(name)[0] for name in layer_names]

    def get_dop_rgb_layer(self):
        return [QgsProject.instance().mapLayersByName(name)[0] for name in ["dop_rgb"]]

    def get_path(self, gid, sub_dir, testset):
        if testset:
            return os.path.join(self.test_base_dir, sub_dir, f"{gid}.png")

        return os.path.join(self.train_base_dir, sub_dir, f"{gid}.png")

    def get_map_settings(self, layers, geom_string, img_size=250):
        settings = QgsMapSettings()
        settings.setLayers(layers)
        settings.setBackgroundColor(QColor(0, 0, 0))
        settings.setOutputSize(QSize(img_size, img_size))

        geom = QgsGeometry.fromWkt(geom_string)
        settings.setExtent(geom.boundingBox())

        return settings

    def render_and_save_image(self, bounding_box, layers, image_size, save_path):
        settings = self.get_map_settings(layers, bounding_box, image_size)
        render = QgsMapRendererSequentialJob(settings)

        render.start()
        render.waitForFinished()
        img = render.renderedImage()
        img.save(save_path, "png")

    def run(self):
        while True:
            try:
                (gid, image_box, label_box, testset) = global_queue.get(block=True, timeout=5)
                image_path = self.get_path(gid, "images", testset)
                label_path = self.get_path(gid, "labels", testset)

                self.render_and_save_image(image_box, self.dop_layer, self.image_size, image_path)
                self.render_and_save_image(label_box, self.segmentation_layer, self.label_size, label_path)
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
        stmt = f"SELECT gid, ST_AsText(geom_image), ST_AsText(geom_label), test_set " \
               f"FROM geom_tiles_{table_suffix} WHERE segment_count>1"
        cur.execute(stmt)

        while True:
            value = cur.fetchone()
            if value is None:
                break

            global_queue.put(tuple(value))


global_queue = queue.Queue(maxsize=1500)
db = psycopg2.connect("dbname='dop10rgbi_nrw' user='postgres' host='localhost' password='root'")

for i in range(5):
    worker = TileRenderWorker(i, "unet", 572, 388)
    QgsApplication.taskManager().addTask(worker)

start = time.time()
add_tiles_to_global_queue(table_suffix="wnet")
global_queue.join()
db.close()
QgsMessageLog.logMessage(f"done - {time.time() - start:.2f}")
