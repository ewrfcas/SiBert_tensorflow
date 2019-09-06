import tensorflow as tf
import blocksparse as bs
import numpy as np


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def split_states(x, n):
    """
    reshape (batch, pixel, state) -> (batch, pixel, head, head_state)
    """
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return tf.reshape(x, new_x_shape)


def split_heads(x, n):
    return tf.transpose(split_states(x, n), [0, 2, 1, 3])


def merge_states(x):
    """
    reshape (batch, pixel, head, head_state) -> (batch, pixel, state)
    """
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)


def merge_heads(x):
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))


def gather_indexes(sequence_tensor, positions, use_sparse=False):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = shape_list(sequence_tensor)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])

    if not use_sparse:
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    else:
        # ipdb.set_trace()
        flat_sequence_tensor = tf.expand_dims(flat_sequence_tensor, 0)
        output_tensor = bs.fancy_gather(flat_sequence_tensor, flat_positions, use_tf=True)

    return output_tensor


def layernorm(x, scope, epsilon=1e-5, relu=False):
    """
    normalize state vector to be zero mean / unit variance + learned scale/shift
    """
    n_state = x.shape[-1].value
    with tf.variable_scope(scope):
        gain = tf.get_variable('gain', [n_state], initializer=tf.constant_initializer(1.0))
        bias = tf.get_variable('bias', [n_state], initializer=tf.constant_initializer(0.0))

        return bs.layer_norm(x, gain, bias, axis=-1, epsilon=epsilon, relu=relu)


def conv1d(x, scope, channel, std=0.02, relu=False, fast_gelu=False, bias=True, float16=False):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        ndims = x.shape.ndims

        # Note: param initializers are not particularly well tuned in this code
        w = tf.get_variable("w", [nx, channel], initializer=create_initializer(initializer_range=std))
        if bias:
            b = tf.get_variable("bias", [channel], initializer=tf.constant_initializer(0.0))
        else:
            b = tf.zeros([channel])

        if float16:
            # We delay weight casting till just before use to minimize memory footprint.
            # In recompute mode these casts are released just after use on forward pass,
            # then remade on the recompute pass.
            with tf.control_dependencies([x.op]):
                # By setting dx_dtype to float16 we prevent useless casting back to fp32 in the backwards pass.
                # Our all-reduce and fused optimizers can accept fp16 natively.
                w = bs.float_cast(w, dtype=tf.float16, dx_dtype=tf.float16)

        # merge context and batch dims for more efficient matmul
        if ndims > 2:
            y_shape = tf.concat([tf.shape(x)[: ndims - 1], [channel]], axis=0)
            x = tf.reshape(x, [-1, nx])

        y = tf.matmul(x, w)

        # avoid atomics in bias grad, but be careful as tf handles temp memory badly in the presense of async ops like all-reduce
        y = bs.bias_relu(y, b, relu=relu, fast_gelu=fast_gelu, atomics=False)

        if ndims > 2:
            y = tf.reshape(y, y_shape)

        return y
