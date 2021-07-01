import glob

import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import display


class ImageLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, image_prefix: str = "image_at_epoch_", gif_name: str = "gif"):
        super(ImageLoggerCallback, self).__init__()
        self.image_prefix = image_prefix
        self.gif_name = gif_name
        self.latent_vector = tf.random.normal([16, 100])

    def on_epoch_end(self, epoch, logs=None):
        generate_and_save_images(
            self.model.generator,
            epoch + 1,
            self.latent_vector,
            prefix=self.image_prefix,
        )
        display.clear_output(wait=True)

    def on_train_end(self, logs=None):
        with imageio.get_writer(f"{self.gif_name}.gif", mode="I") as writer:
            filenames = glob.glob(f"{self.image_prefix}*.png")
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)


def generate_and_save_images(model, epoch, generator_input, prefix):
    predictions = model.predict(generator_input)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    plt.savefig(f"{prefix}{epoch}.png".format(epoch))
