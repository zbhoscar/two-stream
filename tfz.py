import tensorflow as tf


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# conv_var = [[7,7,3,96],[1,2,2,1]]   ksize\strides
def cp_layer(input_var, layer_name, conv_var=None, pool_var=None, relu_bool=True, regu=None, norm=None):
    # with tf.name_scope(layer_name):
    with tf.variable_scope(layer_name):
        if conv_var is not None:
            weights = tf.get_variable('weights', conv_var[0], initializer=tf.truncated_normal_initializer(stddev=0.1))
            # weights = tf.Variable(tf.truncated_normal(conv_var[0], dtype=tf.float32, stddev=1e-1), name='weights')
            # if regu is not None:
            #     tf.add_to_collection('losses', regu(weights))
            biases = tf.get_variable('biases', [conv_var[0][-1]], initializer=tf.constant_initializer(0.1))
            # biases = tf.Variable(tf.constant(0.0, shape=[conv_var[0][-1]], dtype=tf.float32), trainable=True,
            #                      name='biases')
            # variable_summaries(weights)
            # variable_summaries(biases)
            conv = tf.nn.bias_add(tf.nn.conv2d(input_var, weights, conv_var[1], padding='SAME'), biases)
        else:
            conv = input_var
        if relu_bool is True:
            relu = tf.nn.relu(conv)
        else:
            relu = conv
        if pool_var is not None:
            pool = tf.nn.max_pool(relu, ksize=pool_var[0], strides=pool_var[1], padding='SAME')
        else:
            pool = relu
    # tf.summary.histogram(layer_name, relu)
    return pool


def fc_layer(input_var, layer_name, fc_var, dropout=True, regu=None, loss_name='loss'):
    with tf.variable_scope(layer_name):
    #with tf.name_scope(layer_name):
        weights = tf.get_variable('weights', fc_var, initializer=tf.truncated_normal_initializer(stddev=0.1))
        #weights = tf.Variable(tf.truncated_normal(fc_var, dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.get_variable('biases', [fc_var[-1]], initializer=tf.constant_initializer(0.1))
        # biases = tf.Variable(tf.constant(0.0, shape=[fc_var[-1]], dtype=tf.float32), trainable=True, name='biases')
        if regu is not None:
            tf.add_to_collection(loss_name, regu(weights))
        fc = tf.nn.relu(tf.matmul(input_var, weights) + biases)
        if dropout:
            fc = tf.nn.dropout(fc, 0.5)
    return fc

# # conv_var = [[7,7,3,96],[1,2,2,1]]   ksize\strides
# def cp_layer(input_var, layer_name, conv_var=None, pool_var=None, relu_bool=True, dropout=True, regu=None, norm=None):
#     with tf.name_scope(layer_name):
#         if conv_var is not None:
#             weights = tf.Variable(tf.truncated_normal(conv_var[0], dtype=tf.float32, stddev=1e-1), name='weights')
#             if regu is not None:
#                 tf.add_to_collection('losses', regu(weights))
#             biases = tf.Variable(tf.constant(0.0, shape=[conv_var[0][-1]], dtype=tf.float32), trainable=True,
#                                  name='biases')
#             # variable_summaries(weights)
#             # variable_summaries(biases)
#             conv = tf.nn.bias_add(tf.nn.conv2d(input_var, weights, conv_var[1], padding='SAME'), biases)
#         else:
#             conv = input_var
#         if relu_bool is True:
#             relu = tf.nn.relu(conv)
#         else:
#             relu = conv
#         if pool_var is not None:
#             pool = tf.nn.max_pool(relu, ksize=pool_var[0], strides=pool_var[1], padding='SAME')
#         else:
#             pool = relu
#     # tf.summary.histogram(layer_name, relu)
#     return pool


# def fc_layer(input_var, layer_name, fc_var, dropout=True, regu=None, loss_name='loss'):
#     with tf.name_scope(layer_name):
#         weights = tf.Variable(tf.truncated_normal(fc_var, dtype=tf.float32, stddev=1e-1), name='weights')
#         biases = tf.Variable(tf.constant(0.0, shape=[fc_var[-1]], dtype=tf.float32), trainable=True, name='biases')
#         if regu is not None:
#             tf.add_to_collection(loss_name, regu(weights))
#         fc = tf.nn.relu(tf.matmul(input_var, weights) + biases)
#         if dropout:
#             fc = tf.nn.dropout(fc, 0.5)
#     return fc