import tensorflow as tf
import tfz


def inference(input_batch, input_channel_num, class_num, regu, train=True, net_stream='spatial'):
    # conv_var=None, pool_var=None, relu_bool=True, regu=None, norm=None
    cp_net_info = [
        [[[7, 7, input_channel_num, 96], [1, 2, 2, 1]], [[1, 3, 3, 1], [1, 2, 2, 1]], True, regu, None],
        [[[5, 5, 96, 256], [1, 2, 2, 1]], [[1, 3, 3, 1], [1, 2, 2, 1]], True, regu, None],
        [[[3, 3, 256, 512], [1, 1, 1, 1]], None, True, regu, None],
        [[[3, 3, 512, 512], [1, 1, 1, 1]], None, True, regu, None],
        [[[3, 3, 512, 512], [1, 1, 1, 1]], [[1, 3, 3, 1], [1, 2, 2, 1]], True, regu, None],
    ]
    temp = input_batch
    for i in range(len(cp_net_info)):
        cp_name = '%s-layer%d' % (net_stream, i + 1)
        temp = tfz.cp_layer(temp, cp_name, cp_net_info[i][0], cp_net_info[i][1], cp_net_info[i][2],
                            cp_net_info[i][3], cp_net_info[i][4])
    pool_shape = temp.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(temp, [pool_shape[0], nodes])
    # (input_var, layer_name, fc_var, dropout=True, regu=None, loss_name='loss')
    fc_net_info = [[[nodes, 4096], train, regu, net_stream + '_loss'],
                   [[4096, 2048], train, regu, net_stream + '_loss'],
                   [[2048, class_num], False, regu, net_stream + '_loss']]
    for j in range(len(fc_net_info)):
        fc_name = '%s-layer%d' % (net_stream, j + len(cp_net_info) + 1)
        reshaped = tfz.fc_layer(reshaped, fc_name,
                                fc_net_info[j][0], fc_net_info[j][1], fc_net_info[j][2], fc_net_info[j][3])
    return reshaped
