import tensorflow as tf


# import re


def _activation_summary(x, tower_name='tower'):
    # # tensor_name = re.sub('%s_[0-9]*/' % tower_name, '', x.op.name)
    # tensor_name = x.op.name
    # tf.summary.histogram(tensor_name + '/activations', x)
    # tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    return None


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def alex_inference_for_two_stream(images, batch_size, input_chanel, num_class, stream='_spatial'):
    # conv1
    with tf.variable_scope('conv1' + stream) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[7, 7, input_chanel, 96],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # norm1
    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # pool1
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1' + stream)

    # conv2
    with tf.variable_scope('conv2' + stream) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 96, 256],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # norm2
    if stream == '_temporal':
        norm2 = conv2
    else:
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2' + stream)

    # conv3
    with tf.variable_scope('conv3' + stream) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 256, 512],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv3)

    # conv4
    with tf.variable_scope('conv4' + stream) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv4)

    # conv5
    with tf.variable_scope('conv5' + stream) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 512, 512],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv5)

    # pool5
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool5' + stream)

    # local1
    with tf.variable_scope('local1' + stream) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool5, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 4096],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.1))
        local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local1)

    # local2
    with tf.variable_scope('local2' + stream) as scope:
        weights = _variable_with_weight_decay('weights', shape=[4096, 2048],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [2048], tf.constant_initializer(0.1))
        local2 = tf.nn.relu(tf.matmul(local1, weights) + biases, name=scope.name)
        _activation_summary(local2)

    # logits
    with tf.variable_scope('softmax_linear' + stream) as scope:
        weights = _variable_with_weight_decay('weights', [2048, num_class],
                                              stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [num_class],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local2, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear
