import model

import os
import tensorflow as tf

from utils import broadcasted_iou, process_outputs, TbResultsVisualization
from metrics import YoloLoss
from absl import app
from absl import flags
from functools import partial


FLAGS = flags.FLAGS
flags.DEFINE_string("train_dir", "/media/tom/HDD-HARD-DISK-1/datasets/object_detection/", "Path to training directory")  #TODO none
flags.DEFINE_string("test_dir", None, "Path to evaluation directory")
flags.DEFINE_string("logging_dir", "./logs", "Path to log directory for tensorboard and checkpoints")
flags.DEFINE_integer("width", 256, "Images width")
flags.DEFINE_integer("height", 256, "Images height")
flags.DEFINE_integer("n_class", 2, "Number of object classes")
flags.DEFINE_integer("n_epochs", 150, "Number of object classes")
flags.DEFINE_integer("batch_size", 5, "Number of object classes")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")


def read_sample(line):
    """Return image and labels for object detection.
    Retrun:
    image: input image for yolo model,
    labels: label for each yolo detection head (bounding box in pixel location not in image relative position)
    """
    line = tf.strings.split(line)
    image = tf.io.read_file(FLAGS.train_dir + line[0])
    image = tf.image.convert_image_dtype(tf.image.decode_jpeg(image), dtype=tf.float32)
    image_shape = tf.shape(image)

    image = tf.image.resize(image, [FLAGS.width, FLAGS.height])
    labels = tf.strings.to_number(line[1:], tf.float32)

    # transform bounding box coord to image relative position
    labels /= tf.cast(tf.tile([image_shape[0], image_shape[1]], [2]), dtype=tf.float32)
    labels = labels[tf.newaxis, ...]
    return image, labels  # labels in shape n_box, 4


def process_batch(images, labels, grid, anchors):
    """Map function for batch preprocessing. Convert bounding box labels to yolo output shape.
    Args:
        labels: ground true labels (batch_size, n_box, x0, y0, x1, y1),
        grid: number array of grid cell for each output,
        cell_sizes: size of one cell in raw pixel,
    """
    n_outputs = tf.shape(grid)[0]
    labels_shape = tf.shape(labels)  # (batch_size, n_box)
    yolo_labels = tf.TensorArray(tf.float32, size=n_outputs, dynamic_size=False, infer_shape=False)

    xy_center = (labels[..., 2:4] + labels[..., :2]) / 2

    for i in tf.range(n_outputs):
        n_anchors = tf.shape(anchors[i])[0]
        # Find bounding box center xy position

        cell_sizes = tf.cast(1 / grid[i], tf.float32)
        cell_sizes = tf.broadcast_to(cell_sizes[tf.newaxis, tf.newaxis, ...], tf.shape(xy_center))
        # Cell responsible for predictions
        cell_coord = tf.cast(xy_center / cell_sizes, dtype=tf.int32)

        # Set box point to x=0, y=0 for iou calculation
        labels_centered = tf.zeros((labels_shape[0], labels_shape[1], 2))
        labels_centered = tf.concat([labels_centered, labels[..., 2:4] - labels[..., 0:2]], axis=-1)

        # Set anchors point to x=0, y=0 for iou calculation
        anch_centered = tf.zeros((tf.shape(anchors[i])))
        anch_centered = tf.concat([anch_centered, anchors[i]], axis=-1)[tf.newaxis, ...]

        ious = broadcasted_iou(labels_centered, anch_centered)

        max_ious = tf.argmax(ious, axis=-1, output_type=tf.int32)
        batch = tf.zeros((labels_shape[0], grid[i], grid[i], n_anchors, 6))  # Empty batch

        # Create batch, grid(x,y), anchor index
        batch_indexes = tf.range(labels_shape[0])[..., tf.newaxis]
        batch_indexes = tf.broadcast_to(batch_indexes, (labels_shape[0], labels_shape[1]))

        # swap grid coord [x, y] => [y, x]
        indexes = tf.concat([batch_indexes, cell_coord[..., 1], cell_coord[..., 0], max_ious], axis=-1)
        indexes = tf.reshape(indexes, [-1, 4])

        fake_label = tf.ones((labels_shape[0], labels_shape[1], 1))  # TODO dodat label
        fake_label = tf.reshape(fake_label, (-1, 1))

        updates = tf.reshape(labels, [-1, 4])
        # concat box coordinate x0, y0, x1, y1, with obj(1) and class sparse label
        updates = tf.concat([updates, tf.ones((labels_shape[0] * labels_shape[1], 1)), fake_label], axis=-1)
        batch = tf.tensor_scatter_nd_update(batch, indexes, updates)
        yolo_labels = yolo_labels.write(i, batch)

    return images, (yolo_labels.read(0), yolo_labels.read(1), yolo_labels.read(2))


def create_callbacks(log_dir, n_class, anchors, val_images, val_labels):
    process_fn = partial(process_outputs, n_class=n_class, anchors=anchors)
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', mode='min', patience=4, factor=0.1, min_lr=1e-6),
        TbResultsVisualization(log_dir, val_images, val_labels, process_fn)
    ]
    return callbacks


def main(argv):
    input_shape = [FLAGS.width, FLAGS.height, 3]
    anchors = model.yolo_anchors / FLAGS.width
    anchors = tf.gather_nd(anchors, model.yolo_anchor_masks[..., tf.newaxis])
    yolo = model.yolo_v3(input_shape, anchors=anchors, n_class=FLAGS.n_class, training=True)

    grid = tf.Variable([output.shape[1] for output in yolo.outputs], dtype=tf.int32)
    losses = [YoloLoss(input_shape, tf.constant(FLAGS.n_class, tf.int32), anchors[i],
                       name=f"head_{i+1}_loss") for i in range(len(yolo.outputs))]

    dataset = tf.data.TextLineDataset(FLAGS.train_dir + "labels.txt")
    dataset = dataset.map(read_sample)

    if not os.path.exists(FLAGS.logging_dir):
        os.makedirs(FLAGS.logging_dir)

    val_images, val_labels = next(dataset.batch(5).take(1).as_numpy_iterator())
    callbacks = create_callbacks(FLAGS.logging_dir, FLAGS.n_class, anchors, val_images, val_labels)
    del val_images, val_labels

    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.map(lambda img, label: process_batch(img, label, tf.Variable(grid), tf.Variable(anchors))).prefetch(tf.data.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    yolo.compile(optimizer=optimizer, loss=losses, run_eagerly=True) #run_eagerly=True
    yolo.fit(dataset, epochs=FLAGS.n_epochs, callbacks=callbacks)


if __name__ == "__main__":
    app.run(main)
