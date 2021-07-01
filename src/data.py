import tensorflow as tf

from config import (
    ADJUSTED_TOTAL_SAMPLES,
    BATCH_SIZE,
    LABELED_TRAIN_SAMPLES,
    SMALL_BATCH_SIZE,
    UNLABELED_TRAIN_SAMPLES,
    VALIDATION_SAMPLES,
)


def make_dataset_performant(
    dataset: tf.data.Dataset, is_small=False
) -> tf.data.Dataset:
    if is_small:
        return (
            dataset.shuffle(len(dataset))
            .cache()
            .batch(SMALL_BATCH_SIZE, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        return (
            dataset.shuffle(len(dataset))
            .cache()
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32")
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32")

full_train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
full_train_ds = make_dataset_performant(full_train_ds)

x_train = x_train[:ADJUSTED_TOTAL_SAMPLES]
y_train = y_train[:ADJUSTED_TOTAL_SAMPLES]

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5


labeled_train_x, unlabeled_train_x, val_x = tf.split(
    x_train,
    [LABELED_TRAIN_SAMPLES, UNLABELED_TRAIN_SAMPLES, VALIDATION_SAMPLES],
    axis=0,
)
labeled_train_y, unlabeled_train_y, val_y = tf.split(
    y_train,
    [LABELED_TRAIN_SAMPLES, UNLABELED_TRAIN_SAMPLES, VALIDATION_SAMPLES],
    axis=0,
)

labeled_train_ds = tf.data.Dataset.from_tensor_slices(
    (labeled_train_x, labeled_train_y)
)
labeled_train_ds = make_dataset_performant(labeled_train_ds, is_small=True)


unlabeled_train_ds = tf.data.Dataset.from_tensor_slices(
    (unlabeled_train_x, unlabeled_train_y)
)
unlabeled_train_ds = unlabeled_train_ds.map(lambda X, y: X)
unlabeled_train_ds = make_dataset_performant(unlabeled_train_ds)

val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_ds = make_dataset_performant(val_ds)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = make_dataset_performant(test_ds)
