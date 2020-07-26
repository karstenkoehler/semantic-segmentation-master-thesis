import tensorflow as tf


class ArgmaxMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


TF_CUSTOM_METRICS = {
    "ArgmaxMeanIoU": ArgmaxMeanIoU,
}
