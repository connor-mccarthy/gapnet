from typing import Dict, Optional, Tuple, Union

import tensorflow as tf


def get_gaussian_latent_vector(batch_size, input_shape):
    return tf.random.normal(shape=(batch_size, *input_shape))


class GAN(tf.keras.Model):
    def __init__(
        self,
        generator: tf.keras.Model,
        discriminator: tf.keras.Model,
        name: Optional[str] = None,
    ) -> None:
        super(GAN, self).__init__(name=name)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_vector = self.generator.input.get_shape()[1:]

    def compile(
        self,
        g_optimizer: tf.keras.optimizers.Optimizer,
        d_optimizer: tf.keras.optimizers.Optimizer,
        g_loss: tf.keras.losses.Loss,
        d_loss: tf.keras.losses.Loss,
    ) -> None:
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.g_loss = g_loss
        self.d_loss = d_loss

        self.g_bce_tracker = tf.keras.metrics.BinaryCrossentropy(name="g_bin_ce")
        self.d_bce_tracker = tf.keras.metrics.BinaryCrossentropy(name="d_bin_ce")

        self.g_acc_tracker = tf.keras.metrics.BinaryAccuracy(name="g_acc")
        self.d_acc_tracker = tf.keras.metrics.BinaryAccuracy(name="d_acc")

    @property
    def metrics(self):
        return [
            self.g_bce_tracker,
            self.d_bce_tracker,
            self.g_acc_tracker,
            self.d_acc_tracker,
        ]

    def train_step(
        self, real_images: Union[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]
    ) -> Dict[str, float]:
        has_label = isinstance(real_images, tuple) and (len(real_images) == 2)
        if has_label:
            real_images = real_images[0]

        batch_size = tf.shape(real_images)[0]
        latent_vector = get_gaussian_latent_vector(batch_size, self.latent_vector)

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            fake_images = self.generator(latent_vector, training=True)

            fake_probs = self.discriminator(fake_images, training=True)
            real_probs = self.discriminator(real_images, training=True)

            combined_probs = tf.concat([fake_probs, real_probs], axis=0)
            labels = tf.concat(
                [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
            )

            g_loss_val = self.g_loss(tf.ones_like(fake_probs), fake_probs)

            labels += 0.05 * tf.random.uniform(tf.shape(labels))
            d_loss_val = self.d_loss(combined_probs, labels)

        g_gradients = g_tape.gradient(g_loss_val, self.generator.trainable_variables)
        d_gradients = d_tape.gradient(
            d_loss_val, self.discriminator.trainable_variables
        )

        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        self.d_optimizer.apply_gradients(
            zip(d_gradients, self.discriminator.trainable_variables)
        )

        self.g_bce_tracker.update_state(tf.ones_like(fake_probs), fake_probs)
        self.d_bce_tracker.update_state(combined_probs, labels)
        self.g_acc_tracker.update_state(tf.ones_like(fake_probs), fake_probs)
        self.d_acc_tracker.update_state(combined_probs, labels)

        return {metric.name: metric.result() for metric in self.metrics}
