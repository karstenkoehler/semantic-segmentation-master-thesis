import tensorflow as tf


class ArgmaxMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def weighted_categorical_crossentropy(class_weights):
    class_weights = tf.constant(class_weights)

    def wcce(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=class_weights)

    return wcce


TF_CUSTOM_METRICS = {
    "ArgmaxMeanIoU": ArgmaxMeanIoU,
}
