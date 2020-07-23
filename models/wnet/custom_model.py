import os

import tensorflow as tf


class WNetModel(tf.keras.models.Model):
    def __init__(self, encoder, full_model):
        super().__init__()
        self.encoder = encoder
        self.full_model = full_model

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.encoder(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.encoder.compiled_loss(y, y_pred, regularization_losses=self.encoder.losses)

        # Compute gradients
        trainable_vars = self.encoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.encoder.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.encoder.compiled_metrics.update_state(y, y_pred)

        with tf.GradientTape() as tape:
            y_pred = self.full_model(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.full_model.compiled_loss(y, y_pred, regularization_losses=self.full_model.losses)

        # Compute gradients
        trainable_vars = self.full_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.full_model.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.full_model.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.full_model.metrics}
