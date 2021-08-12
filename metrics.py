import tensorflow as tf
import utils


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, img_size, n_class, anchors, name=None, threshold=0.5):
        self.img_size = img_size
        self.threshold = threshold
        self.n_class = n_class
        self.anchors = anchors
        self.cls_loss = 0
        self.coord_loss = 0
        self.obj_loss = 0

        super(YoloLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        """Compute YOLO loss.
        Args:
            y_true: ground true (batch_size, x, y, n_anchors, (x0, y0, x1, y1, obj, n_class))
            y_pred: output from yolo (batch_size, x, y, n_anchors, 5+n_class),
        """
        shape = tf.shape(y_pred)[:-1]
        y_true = tf.cast(y_true, y_pred.dtype)  # cast y_true to float
        y_pred_coord, pred_obj, pred_cls = utils.process_yolo_output(y_pred, self.anchors)  # find pred relative pos
        y_pred_coord = tf.reshape(y_pred_coord, (shape[0], -1, 4))  # x0, x1 point coordinates
        y_true_coord, obj, y_class = tf.split(y_true, (4, 1, 1), axis=-1)

        # calculate iou between y_true and y_pred
        true_box_flat = tf.boolean_mask(y_true_coord, tf.cast(tf.squeeze(obj, axis=-1), tf.bool))
        ious = utils.broadcasted_iou(utils.coordinates_to_points(y_pred_coord), true_box_flat)

        noobj = tf.math.greater(ious, self.threshold)
        noobj = tf.reduce_any(noobj, axis=-1)
        noobj = tf.reshape(noobj, (-1, shape[1], shape[1], shape[-1]))  # (batch, x, y, n_anchor)

        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        y_true_coord = utils.points_to_coordinates(y_true_coord)    # from x0, x1 to x_center, y_center, w, h
        obj = tf.squeeze(obj)
        true_wh = y_true_coord[..., 2:4]
        pred_wh = y_pred[..., 2:4]
        scale_factor = 2 - true_wh[..., 0] * true_wh[..., 1]  # to scale loss for small bounding box

        # noobj = 1 if not obj and iou < threshold otherwise 0
        noobj = tf.math.logical_not(tf.math.logical_or(tf.cast(obj, tf.bool), noobj))
        noobj = tf.cast(noobj, obj.dtype)

        # convert img relative xy position to cell relative position
        grid_xy = tf.meshgrid(tf.range(shape[1]), tf.range(shape[2]))
        grid_xy = tf.expand_dims(tf.stack(grid_xy, axis=-1), axis=2)

        true_xy = y_true_coord[..., :2] * tf.cast(shape[1], tf.float32) - tf.cast(grid_xy, tf.float32)
        pred_xy = tf.nn.sigmoid(y_pred[..., :2])

        true_wh = tf.math.log(true_wh / self.anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        # bounding box loss
        # reduction sum over [x,y],[w,h] coordinate
        xy_loss = obj * scale_factor * tf.reduce_sum(tf.math.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj * scale_factor * tf.reduce_sum(tf.math.square(true_wh - pred_wh), axis=-1)

        # confidence loss
        obj_loss = obj * tf.square(obj - pred_obj)
        noobj_loss = noobj * tf.square(obj - pred_obj)

        # classification loss
        cls = y_true[..., 5]  # True labels
        cls = tf.one_hot(tf.cast(cls, dtype=tf.int32), self.n_class)
        cls_loss = obj * tf.reduce_sum(tf.square(cls - pred_cls), axis=-1)  # reduce sum over classes

        # reduction sum over x, y, anchors
        xy_loss = tf.reduce_sum(xy_loss, [1, 2, 3])
        wh_loss = tf.reduce_sum(wh_loss, [1, 2, 3])
        obj_loss = tf.reduce_sum(obj_loss, [1, 2, 3])
        noobj_loss = tf.reduce_sum(noobj_loss, [1, 2, 3])
        cls_loss = tf.reduce_sum(cls_loss, [1, 2, 3])
        self.xy_loss = tf.reduce_mean(xy_loss)
        self.wh_loss = tf.reduce_mean(wh_loss)

        return tf.reduce_mean(xy_loss + wh_loss + obj_loss + noobj_loss + cls_loss)


