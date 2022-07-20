import io

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def broadcasted_iou(preds, bbox):
    """"Computes intersection over union for predicted bounding box
    Args:
        preds: bounding box predicted by model (batch_size, n_box, x0+y0+x1+y1)
        bbox: ground true bounding box (..., x0+y0+x1+y1)
    """
    preds = tf.expand_dims(preds, -2)
    bbox = tf.expand_dims(bbox, 0)
    # Find broadcast shape
    broadcast_shape = tf.broadcast_dynamic_shape(tf.shape(preds), tf.shape(bbox))

    preds = tf.broadcast_to(preds, broadcast_shape)
    bbox = tf.broadcast_to(bbox, broadcast_shape)

    # find intersection rectangle
    # if x0 < x1 and y0 < y1 then they not overlap
    intersection_width = tf.maximum(tf.minimum(preds[..., 2], bbox[..., 2]) -
                       tf.maximum(preds[..., 0], bbox[..., 0]), 0)
    intersection_height = tf.maximum(tf.minimum(preds[..., 3], bbox[..., 3]) -
                       tf.maximum(preds[..., 1], bbox[..., 1]), 0)

    intersection = intersection_width * intersection_height
    preds_content = (preds[..., 2] - preds[..., 0]) * (preds[..., 3] - preds[..., 1])
    bbox_content = (bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1])
    iou = intersection / (preds_content + bbox_content - intersection)

    return iou


def process_yolo_output(preds, anchors):
    """Transform yolo predictions to bounding box
    Args:
        preds: predicted bounding box (batch, x, y, n_anchors, bx+by+bh+bw)
        anchors: predefined anchor box,
        image_shape: input image shape
    Returns:
        bounding box coordinates [x0, x1]
    """
    preds_shape = tf.shape(preds)[1:-1]  # x, y, n_anchors

    delta_x, delta_y = tf.meshgrid(tf.range(preds_shape[0]), tf.range(preds_shape[1]))
    delta_xy = tf.concat([delta_x[..., tf.newaxis], delta_y[..., tf.newaxis]], axis=-1)

    delta_xy = tf.broadcast_to(delta_xy[tf.newaxis, :, :, tf.newaxis, :],
                               (1, preds_shape[0], preds_shape[1], preds_shape[2], 2))
    delta_xy = tf.cast(delta_xy, dtype=preds.dtype)

    # Relative position of the predicted bounding box center point
    center_xy = (tf.nn.sigmoid(preds[..., 0:2]) + delta_xy) / tf.cast(preds_shape[0], tf.float32)

    # Broadcast anchors to (x,y,n_anchors, 2(width, height))
    anchors = tf.cast(tf.broadcast_to(anchors[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                                      (1, preds_shape[0], preds_shape[1], preds_shape[2], 2)), dtype=tf.float32)
    pred_wh = anchors * tf.math.exp(preds[..., 2:4])

    obj = tf.nn.sigmoid(preds[..., 4])
    cls = tf.nn.sigmoid(preds[..., 5:])

    # (batch, x, y, n_anchors, x_center+y_center+width+height)
    return tf.concat((center_xy, pred_wh), axis=-1), obj, cls


def coordinates_to_points(coordinate):
    """Transform x, y, w, h coordinate to x0, x1 points.
    Args:
        coordinate: output with bounding box center, width and height (..., 4)
    Return:
        Bounding box point x0 and x1 (..., 4)
    """
    center_xy, box_wh = tf.split(coordinate, (2, 2), axis=-1)

    xy0 = center_xy - box_wh / 2
    xy1 = center_xy + box_wh / 2
    return tf.concat([xy0, xy1], axis=-1)


def points_to_coordinates(points):
    """Transform x0, x1 bounding box points to x, y, w, h coordinates.
    Args:
        points: x0, y0, x1, y1 shape (..., 4)
    Return:
        Bounding box coordinate x_center, y_center, w, h (..., 4)
    """
    x0, x1 = tf.split(points, (2, 2), axis=-1)

    box_wh = x1 - x0
    xy_center = (x0 + x1) / 2

    return tf.concat([xy_center, box_wh], axis=-1)


@tf.function
def process_outputs(preds, n_class, anchors, threshold=0.5, score_threshold=0.5, max_boxes=2000):
    """"
    Args:
        preds: prediction from object detection model (n_outputs, batch_dimension, x, y, n_anchors*(bx+by+bw+bh+obj+n_class)),
        n_class: number of classes,
        anchors: predefined anchors shape: (n_outputs, n_anchors, 2 (w, h)),
        image_shape: size of original image for rescaling bounding box
        threshold: threshold for NMS,
        max_boxes: maximum number of boxes
    Returns:
        The most accurate predicted bounding box.
    """
    yolo_scores = []
    yolo_boxes = []

    n_anchors = tf.shape(anchors)
    for i, pred in enumerate(preds):
        anchor = anchors[i]
        shape = tf.shape(pred)
        n_boxes = tf.reduce_prod([shape[1], shape[2], n_anchors[1]])

        bbox, obj, scores = process_yolo_output(pred, anchor)
        bbox_xy, bbox_wh = tf.split(bbox, (2, 2), -1)

        bbox = tf.concat([bbox_xy, bbox_wh], axis=-1)
        bbox = coordinates_to_points(bbox)                     # (xy_center, width, height) => [x0, x1]

        bbox = tf.reshape(bbox, (-1, n_boxes, 1, 4))           # reshape to batch, n_boxes, 1, 4(y0, x0, y1, x1)

        scores = obj[..., tf.newaxis] * scores                  # multiply obj with class scores
        scores = tf.reshape(scores, (-1, n_boxes, n_class))    # reshape to batch, n_boxes, n_class

        yolo_scores.append(scores)
        yolo_boxes.append(bbox)

    yolo_scores = tf.concat(yolo_scores, axis=1)
    yolo_boxes = tf.concat(yolo_boxes, axis=1)

    box_index = tf.image.combined_non_max_suppression(yolo_boxes, yolo_scores,
                                                      iou_threshold=threshold,
                                                      max_total_size=max_boxes,
                                                      score_threshold=score_threshold,
                                                      max_output_size_per_class=100,
                                                      clip_boxes=False)

    return box_index


# =========================================== Custom callback =========================================================
class TbResultsVisualization(tf.keras.callbacks.Callback):
    """Callback to visualize model predictions in tensorboard every n epoch.
             Args:
                 logdir: path to logging directory,
                 img_test: batch of images to display,
                 test_labels: bround true bounding box
    """
    def __init__(self, log_dir, img_test, test_labels, process_fn, n_epoch=1):
        self.log_dir = log_dir
        self.img_test = img_test
        self.n_epoch = n_epoch
        self.process_fn = process_fn
        self.writer = tf.summary.create_file_writer(log_dir)
        self.img_shape = tf.shape(img_test)
        super(TbResultsVisualization, self).__init__()

        # initialize matplotlib figure with original images
        self.fig = plt.figure(figsize=(12, max(12 // self.img_shape[0] * 2, 1)))
        gs = self.fig.add_gridspec(2, self.img_shape[0].numpy(), hspace=0.2, wspace=0.1)
        self.axs = gs.subplots(sharex='col', sharey='row')
        for i, img in enumerate(self.img_test):            # Plot validation images
            self.axs[0, i].set_title('Ground true')
            self.axs[1, i].set_title('Prediction')
            self.axs[0, i].axis('off')
            self.axs[1, i].axis('off')
            self.axs[0, i].imshow(img)
            self.axs[1, i].imshow(img)

        for i, image_gts in enumerate(test_labels):   # Plot ground true rectangles for each images and gt
            for gt in image_gts:
                gt = gt * tf.cast(tf.tile(self.img_shape[1:3], [2]), tf.float32)
                rect = patches.Rectangle((gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1], lw=1, ec='r', fc='none')
                self.axs[0, i].add_patch(rect)

    def on_epoch_begin(self, epoch, logs=None):
        lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        with self.writer.as_default():
            tf.summary.scalar("learning rate", lr, step=epoch)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.n_epoch != 0:
            return

        for i, img in enumerate(self.img_test):         # Clear figure predictions
            self.axs[1, i].clear()
            self.axs[1, i].set_title('Prediction')
            self.axs[1, i].axis('off')
            self.axs[1, i].imshow(img)

        preds = self.model(self.img_test)
        preds = self.process_fn(preds)
        self.plot_rectangle(preds)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        image = tf.image.decode_png(buf.getvalue(), channels=3)
        image = tf.expand_dims(image, 0)
        with self.writer.as_default():
            tf.summary.image("Validation test", image, step=epoch)

    def plot_rectangle(self, pred):
        rec_coord = pred[0] * tf.cast(tf.tile([self.img_shape[1], self.img_shape[2]], [2]), tf.float32)
        rec_coord = rec_coord[..., :4]
        x0 = rec_coord[..., :2]
        wh = rec_coord[..., 2:4] - x0

        for img, n_rec in enumerate(pred[3]):   # take number of valid rectangles for each image
            for j in range(n_rec):              # for each valid rectangle
                rect = patches.Rectangle((x0[img, j, 0], x0[img, j, 1]),
                                         wh[img, j, 0], wh[img, j, 1], lw=1, ec='r', fc='none')
                self.axs[1, img].add_patch(rect)
