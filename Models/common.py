import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn
import numpy as np


DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', relu=True, groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                 value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    if relu:
        bias = tf.nn.relu(bias, name=scope.name)

    return bias


def fc(x, num_in, num_out, name,  batch_norm=False, activation='', is_training=True):
    if batch_norm and isinstance(is_training, bool):
        print("WARNING: is_training NOT SET - " + name)

    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
    if batch_norm:
        # Apply batch normalization
        act = tf.layers.batch_normalization(act, training=is_training)
    
    if activation is 'relu':
        # Apply ReLu non linearity
        act = tf.nn.relu(act)
    elif activation is 'elu':
        # Apply elu non linearity
        act = tf.nn.elu(act)
    elif activation is 'sigmoid':
        # Apply sigmoid non linearity
        act = tf.nn.sigmoid(act)
    elif activation is 'tanh':
        # Apply tanh non linearity
        act = tf.nn.tanh(act)
    elif activation is 'softmax':
        # Apply softmax non linearity
        act = tf.nn.softmax(act)
    
    return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob, name):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob, name=name)
    
def get_euc_err(pred, gt):
    batch_size = pred.shape[0]
    assert batch_size == gt.shape[0]
    if len(pred.shape) == 2:
        pred = pred.reshape((batch_size, -1, 3))
    if len(gt.shape) == 2:
        gt = gt.reshape((batch_size, -1, 3))
    return np.mean(np.sqrt(np.sum((pred - gt)**2, -1)))