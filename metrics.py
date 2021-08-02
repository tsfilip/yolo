import tensorflow as tf
import utils


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, img_size, n_class, anchors, name=None, threshold=0.5, obj_lambda=1, noobj_lambda=1, coord_lambda=1):
        self.img_size = img_size
        self.threshold = threshold
        self.obj_lambda = obj_lambda
        self.noobj_lambda = noobj_lambda
        self.coord_lambda = coord_lambda
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
        y_pred_coord, pred_obj, pred_cls = utils.process_yolo_output(y_pred, self.anchors, self.img_size)
        y_pred_coord = tf.reshape(y_pred_coord, (shape[0], -1, 4))  # x0, x1 point coordinates
        y_true_coord = tf.reshape(y_true[..., 0:4], (shape[0], -1, 4))

        # calculate iou between y_true and y_pred
        ious = utils.broadcasted_iou(utils.coordinates_to_points(y_pred_coord), y_true_coord)
        noobj = tf.math.greater(ious, self.threshold)
        noobj = tf.reduce_any(noobj, axis=-1)
        noobj = tf.reshape(noobj, (-1, shape[1], shape[1], shape[-1]))  # (batch, x, y, n_anchor)

        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        y_true_coord, obj, y_class = tf.split(y_true, (4, 1, 1), axis=-1)
        y_true_coord = utils.points_to_coordinates(y_true_coord)    # from x0, x1 to x_center, y_center, w, h
        y_pred_coord = tf.reshape(y_pred_coord, (-1, shape[1], shape[1], shape[-1], 4))
        obj = tf.squeeze(obj)

        # noobj = 1 if not obj and iou < threshold otherwise 0
        noobj = tf.math.logical_not(tf.math.logical_or(tf.cast(obj, tf.bool), noobj))
        noobj = tf.cast(noobj, obj.dtype)

        true_wh_scaled = tf.math.log(y_true_coord[..., 2:4] / self.anchors)
        true_wh_scaled = tf.where(tf.math.is_inf(true_wh_scaled), tf.zeros_like(true_wh_scaled), true_wh_scaled)
        true_wh_scaled = tf.math.sqrt(true_wh_scaled)  # scale width and height
        pred_wh_scaled = tf.math.sqrt(y_pred_coord[..., 2:4])

        strides = self.img_size[0] / shape[1], self.img_size[1] / shape[2]
        true_xy_scaled = tf.math.floormod(y_true_coord[..., :2], strides) / strides
        pred_xy_scaled = tf.nn.sigmoid(y_pred[..., :2])

        # bounding box loss
        # reduction sum over [x,y],[w,h] coordinate
        xy_loss = self.coord_lambda * obj * tf.reduce_sum(tf.math.square(true_xy_scaled - pred_xy_scaled), axis=-1)       # TODO spatne not x_coord, y_coord ale x, y porovnam tx true a tx pred
        wh_loss = self.coord_lambda * obj * tf.reduce_sum(tf.math.square(true_wh_scaled - pred_wh_scaled), axis=-1)

        # confidence loss
        obj_loss = self.obj_lambda * obj * tf.square(obj - pred_obj)
        noobj_loss = self.noobj_lambda * noobj * tf.square(obj - pred_obj)

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
        self.cls_loss = tf.reduce_mean(cls_loss)
        self.coord_loss = tf.reduce_mean(xy_loss + wh_loss)
        self.obj_loss = tf.reduce_mean(obj_loss + noobj_loss)

        return tf.reduce_mean(xy_loss + wh_loss + obj_loss + noobj_loss + cls_loss)


