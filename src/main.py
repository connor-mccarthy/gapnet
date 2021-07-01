import random

import numpy as np
import tensorflow as tf

from config import RANDOM_SEED

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import os

import colorama
import matplotlib.pyplot as plt
from colorama import Fore
from plot_keras_history import plot_history
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN

from config import IMAGE_SHAPE
from config import OUTPUT_CLASSES as MNIST_OUTPUT_CLASSES
from config import SAVED_MODEL_DIR, Z_DIM
from custom_callbacks import ImageLoggerCallback
from data import labeled_train_ds, test_ds, unlabeled_train_ds, val_ds
from gan import GAN
from model_components import (
    attach_classifier_to_model,
    make_discriminator,
    make_generator,
)

colorama.init(autoreset=True)


def fit_model(model: tf.keras.models.Model, use_fast_optimizer=True) -> None:
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    epochs = 1000
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(),
    ]
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        TerminateOnNaN(),
    ]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    compile_kwargs = dict(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    fit_kwargs = dict(
        callbacks=callbacks,
        epochs=epochs,
    )

    model.compile(**compile_kwargs)
    history = model.fit(labeled_train_ds, validation_data=val_ds, **fit_kwargs)
    results = model.evaluate(test_ds)
    print()
    print(Fore.GREEN + "Test set results:")
    for metric, result in zip(model.metrics, results):
        print(Fore.BLUE + f" * {metric.name}:", result)
        print()
    plot_history(history)
    plt.draw()
    plt.pause(0.001)


def fit_baseline_model() -> None:
    discriminator = make_discriminator(IMAGE_SHAPE)
    headless_discriminator = discriminator.get_layer("headless_model")
    classification_layer = tf.keras.layers.Dense(
        MNIST_OUTPUT_CLASSES, activation="softmax"
    )
    baseline_classifier = attach_classifier_to_model(
        headless_discriminator, classification_layer
    )
    fit_model(baseline_classifier, use_fast_optimizer=True)


def fit_gan() -> None:
    # first train the GAN
    generator = make_generator((Z_DIM,))
    discriminator = make_discriminator(IMAGE_SHAPE)
    gan = GAN(generator=generator, discriminator=discriminator, name="gan")

    gan_path = os.path.join(SAVED_MODEL_DIR, "gan")

    try:
        gan.load_weights(gan_path)
    except tf.errors.NotFoundError:
        print(f"Weights saved to {gan_path} not found. Training model.")
        g_optim = tf.keras.optimizers.Adam(1e-5)
        d_optim = tf.keras.optimizers.Adam(1e-4)
        d_loss = tf.keras.losses.BinaryCrossentropy()
        g_loss = tf.keras.losses.BinaryCrossentropy()
        gan.compile(
            g_optimizer=g_optim, d_optimizer=d_optim, g_loss=g_loss, d_loss=d_loss
        )

        gan.fit(
            unlabeled_train_ds,
            epochs=50,
            callbacks=[
                ImageLoggerCallback("images/image_at_epoch_", f"images/{gan.name}_gif"),
                tf.keras.callbacks.TerminateOnNaN(),
            ],
        )

        gan.save_weights(gan_path)
        print(f"Saved {gan.name} weights to {gan_path}.")

    classification_layer = tf.keras.layers.Softmax(MNIST_OUTPUT_CLASSES)

    # now train the actual model
    discriminator = gan.discriminator
    headless_discriminator = discriminator.get_layer("headless_model")
    classification_layer = tf.keras.layers.Dense(
        MNIST_OUTPUT_CLASSES, activation="softmax"
    )
    baseline_classifier = attach_classifier_to_model(
        headless_discriminator, classification_layer
    )
    fit_model(baseline_classifier)


def main() -> None:
    fit_baseline_model()
    fit_gan()


if __name__ == "__main__":
    main()
