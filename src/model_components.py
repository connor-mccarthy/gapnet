from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers


def make_discriminator(input_shape: Tuple[int, ...]) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            layers.Conv2D(64, 5, strides=2, padding="same", input_shape=input_shape),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Conv2D(128, 5, strides=2, padding="same"),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Flatten(name="headless_model"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )


def make_generator(input_shape: Tuple[int, ...]) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            layers.Dense(7 * 7 * 256, use_bias=False, input_shape=input_shape),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape((7, 7, 256)),
            layers.Conv2DTranspose(128, 5, strides=1, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(
                1, 5, strides=2, padding="same", use_bias=False, activation="tanh"
            ),
        ]
    )


def attach_classifier_to_model(
    headless_model: tf.keras.Model,
    classification_layer: tf.keras.layers.Layer,
) -> tf.keras.models.Model:
    return tf.keras.Sequential([headless_model, classification_layer])
