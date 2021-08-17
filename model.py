import tensorflow as tf
import numpy as np

from utils import process_outputs
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Lambda, \
    Concatenate, BatchNormalization, Activation, Add


yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)], np.float32)

yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def detection_layer(input, n_filters, kernel_size, strides=1):
    x = Conv2D(n_filters,
               kernel_size=kernel_size,
               strides=strides,
               activation=None,
               use_bias=False,
               padding='same',
               kernel_regularizer=l2(5e-5))(input)
    x = BatchNormalization()(x)
    output = Activation(tf.nn.leaky_relu)(x)
    return output


def residual_unit(input, n_filters):
    x = detection_layer(input, n_filters // 2, kernel_size=(1, 1))
    x = detection_layer(x, n_filters, kernel_size=(3, 3))
    output = Add()([input, x])
    return output


def darknet53(input_shape, n_filters_origin=32, blocks_repetitions=None, n_outputs=3):
    """Create darknet model.
    Args:
    input_shape: W x H x CH of input images,
    n_filters_origin: number of filters in first conv layer,
    blocks_repetitions: list of residual unit repetitions for each block (original darknet 53 has [1, 2, 8, 8, 4]),
    n_outputs: number of model outputs - len(blocks_repetitions) must be greater or equal than n_outputs)
    """
    assert len(blocks_repetitions) >= n_outputs, "Number of blocks must be greater or equal than number of outputs."

    outputs = []
    input = Input(shape=input_shape)
    x = detection_layer(input, n_filters_origin, kernel_size=(3, 3))

    for i in range(len(blocks_repetitions)):
        n_filters = n_filters_origin * 2 ** (i + 1)
        x = detection_layer(x, n_filters, kernel_size=(3, 3), strides=2)
        for _ in range(blocks_repetitions[i]):
            x = residual_unit(x, n_filters)
        outputs.append(x)

    outputs = outputs[-n_outputs:]
    model = tf.keras.Model(inputs=input, outputs=outputs, name="darknet53")
    return model


def convolutional_set(input, n_filters):
    x = detection_layer(input, n_filters // 2, kernel_size=(1, 1))
    x = detection_layer(x, n_filters, kernel_size=(3, 3))
    x = detection_layer(x, n_filters // 2, kernel_size=(1, 1))
    x = detection_layer(x, n_filters, kernel_size=(3, 3))
    x = detection_layer(x, n_filters // 2, kernel_size=(1, 1))
    return x


def yolo_branch(cnn_input, branch_input, model_name):
    """
    Single branch of yolo model architecture.
    Branch concat the cnn_input with branch_input if exist and then process them through cnn layers.
    Args:
    cnn_input: input from backbone model default darknet53,
    branch_input: input from another down sample branch,
    n_filters: number of filters for output cnn (n_anchors * 5 + number of class)
    """
    input_shape = cnn_input.shape[1:]
    if branch_input is not None:
        inputs = Input(input_shape), Input(branch_input.shape[1:])
        x = detection_layer(inputs[1], branch_input.shape[-1] // 2, kernel_size=(1, 1))
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=-1)([x, inputs[0]])
        cnn_input = [cnn_input, branch_input]
    else:
        x = inputs = Input(input_shape)

    x = convolutional_set(x, input_shape[-1])
    return tf.keras.Model(inputs=inputs, outputs=x, name=model_name)(cnn_input)


def yolo_output(cnn_input, n_anchors, output_filters, model_name):
    shape = cnn_input.shape[1:]
    input = Input(shape)
    x = detection_layer(input, n_filters=shape[-1] * 2, kernel_size=(3, 3))
    x = Conv2D(n_anchors * output_filters, kernel_size=(1, 1), kernel_regularizer=l2(5e-5))(x)  # (x_center + y_center + width + height + obj + n_class)
    output = Lambda(lambda output: tf.reshape(output, (-1, shape[0], shape[1], n_anchors, output_filters)))(x)
    return tf.keras.Model(inputs=input, outputs=output, name=model_name)(cnn_input)


def yolo_v3(input_shape, n_class, anchors, anchor_mask=yolo_anchor_masks, training=False, name="yolo_v3"):
    """Crete and return yolo v3 model.
    Args:
        input_shape: size of input images,
        anchors: predefined anchors for each yolo branch (n_outputs, n_anchors, 2[width, height]),
        anchor_mask: mask for each branch anchors (n_branches, n_anchors),
        n_class: number of classes,
        training: flag if true convert yolo output to coordinate and perform nms,
        name: model name
    """
    output_filters = (5 + n_class)   # n_filters = n_anchors * (5 => (x, y, w, h, obj) + n_class)
    outputs = []
    input = Input(shape=input_shape)
    backbone = darknet53(input_shape, blocks_repetitions=[1, 2, 8, 8, 4], n_outputs=3)
    model_outputs = backbone(input)
    branch_output = None
    for i, output in enumerate(model_outputs[::-1]):
        n_anchors = len(anchor_mask[i])
        branch_output = yolo_branch(output, branch_output, f"yolo_conv_{i}")  #n_anchors * output_filters
        final_output = yolo_output(branch_output, n_anchors, output_filters, f"yolo_output_{i}")
        outputs.append(final_output)

    if training:
        return tf.keras.Model(inputs=input, outputs=outputs, name=name)

    output = Lambda(lambda x: process_outputs(x, n_class, anchors))(outputs)
    return tf.keras.Model(inputs=input, outputs=output, name=name)
