"""Common modules to leaders CNN.

Downsampling, sub-filters...
"""

import math
import numpy as np
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages


# ==============================================================================
# conv2d leaders.
# ==============================================================================
@add_arg_scope
def conv2d_leaders(inputs,
                   num_outputs,
                   kernel_size,
                   rates=[],
                   stride=1,
                   padding='SAME',
                   activation_fn=nn.relu,
                   normalizer_fn=None,
                   normalizer_params=None,
                   weights_initializer=initializers.xavier_initializer(),
                   weights_regularizer=None,
                   biases_initializer=init_ops.zeros_initializer,
                   biases_regularizer=None,
                   reuse=None,
                   variables_collections=None,
                   outputs_collections=None,
                   trainable=True,
                   scope=None,):
    """Adds a 2D convolution followed by an optional batch_norm layer.
    `convolution2d` creates a variable called `weights`, representing the
    convolutional kernel, that is convolved with the `inputs` to produce a
    `Tensor` of activations. If a `normalizer_fn` is provided (such as
    `batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
    None and a `biases_initializer` is provided then a `biases` variable would be
    created and added the activations. Finally, if `activation_fn` is not `None`,
    it is applied to the activations as well.
    Performs a'trous convolution with input stride equal to rate if rate is
    greater than one.
    Args:
        inputs: a 4-D tensor  `[batch_size, height, width, channels]`.
        num_outputs: integer, the number of output filters.
        kernel_size: a list of length 2 `[kernel_height, kernel_width]` of
          of the filters. Can be an int if both values are the same.
        stride: a list of length 2 `[stride_height, stride_width]`.
          Can be an int if both strides are the same. Note that presently
          both strides must have the same value.
        padding: one of `VALID` or `SAME`.
        rate: integer. If less than or equal to 1, a standard convolution is used.
          If greater than 1, than the a'trous convolution is applied and `stride`
          must be set to 1.
        activation_fn: activation function.
        normalizer_fn: normalization function to use instead of `biases`. If
          `normalize_fn` is provided then `biases_initializer` and
          `biases_regularizer` are ignored and `biases` are not created nor added.
        normalizer_params: normalization function parameters.
        weights_initializer: An initializer for the weights.
        weights_regularizer: Optional regularizer for the weights.
        biases_initializer: An initializer for the biases. If None skip biases.
        biases_regularizer: Optional regularizer for the biases.
        reuse: whether or not the layer and its variables should be reused. To be
          able to reuse the layer scope must be given.
        variables_collections: optional list of collections for all the variables or
          a dictionay containing a different list of collection per variable.
        outputs_collections: collection to add the outputs.
        trainable: If `True` also add variables to the graph collection
          `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        scope: Optional scope for `variable_op_scope`.
    Returns:
        a tensor representing the output of the operation.
    Raises:
        ValueError: if both 'rate' and `stride` are larger than one.
    """
    with variable_scope.variable_scope(scope, 'Conv', [inputs],
                                       reuse=reuse) as sc:

        inputs = ops.convert_to_tensor(inputs)
        dtype = inputs.dtype.base_dtype
        # inshape = tf.shape(inputs)

        # Leading kernel size.
        kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
        stride_h, stride_w = utils.two_element_tuple(stride)
        num_filters_in = utils.last_dimension(inputs.get_shape(), min_rank=4)

        # Weights variable.
        weights_shape = [kernel_h, kernel_w,
                         num_filters_in, num_outputs]
        weights_collections = utils.get_variable_collections(
            variables_collections, 'weights')
        weights = variables.model_variable('weights',
                                           shape=weights_shape,
                                           dtype=dtype,
                                           initializer=weights_initializer,
                                           regularizer=weights_regularizer,
                                           collections=weights_collections,
                                           trainable=trainable)
        # Bias variable.
        biases = None
        if biases_initializer is not None:
            biases_collections = utils.get_variable_collections(
                variables_collections, 'biases')
            biases = variables.model_variable('biases',
                                              shape=[num_outputs, ],
                                              dtype=dtype,
                                              initializer=biases_initializer,
                                              regularizer=biases_regularizer,
                                              collections=biases_collections,
                                              trainable=trainable)

        # Convolution at different scales.
        outputs_pool = []
        for rate in rates:
            if rate > 1:
                conv = nn.atrous_conv2d(inputs, weights, rate, padding='SAME')
            else:
                conv = nn.conv2d(inputs, weights, [1, 1, 1, 1], padding='SAME')
            outputs_pool.append(conv)
        # 'Pooling' at different scales. A bit hacky. Use of concat + max_pool?
        outputs = None
        outputs_pool.reverse()
        for node in outputs_pool:
            if outputs is None:
                outputs = node
            else:
                outputs = tf.maximum(outputs, node)
        # Add bias?
        if biases is not None:
            outputs = tf.nn.bias_add(outputs, biases)

        # Fix padding and stride. A bit hacky too and not so efficient!
        if padding == 'VALID' or stride_h > 1 or stride_w > 1:
            padfilter = np.zeros(shape=(kernel_h, kernel_w, num_filters_in, 1),
                                 dtype=dtype)
            x = (kernel_h - 1) / 2
            y = (kernel_w - 1) / 2
            padfilter[x, y, :, 0] = 1.
            outputs = tf.nn.depthwise_conv2d(outputs, padfilter,
                                             [1, stride_h, stride_w, 1],
                                             padding=padding)

        # Batch norm and activation...
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            if biases_initializer is not None:
                biases_collections = utils.get_variable_collections(
                    variables_collections, 'biases')
                biases = variables.model_variable('biases',
                                                  shape=[num_outputs, ],
                                                  dtype=dtype,
                                                  initializer=biases_initializer,
                                                  regularizer=biases_regularizer,
                                                  collections=biases_collections,
                                                  trainable=trainable)
                outputs = nn.bias_add(outputs, biases)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections,
                                           sc.name, outputs)
