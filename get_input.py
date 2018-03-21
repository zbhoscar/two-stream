def get_2stream_input(tfrecord_list, batch_size, of_stack_num, in_net_size, for_eval=False):
    import tensorflow as tf
    import datasets_reader as dr
    from multiprocessing import cpu_count

    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(tfrecord_list, shuffle=not for_eval)  # , num_epochs=1)
    # filename_queue = tf.train.string_input_producer(
    #    [r'D:\Desktop\DEMO_FILE\UCF101_tfrecords_10of\data.tfrecords-v_ApplyEyeMakeup_g02_c04'])
    _, serialized_example = reader.read(filename_queue)
    read_tfrecord = dr.make_feature_eval_str(of_stack_num)
    features = eval(read_tfrecord)
    label = features['label']
    # decode image
    image = tf.image.decode_jpeg(features['image_raw'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, in_net_size, method=0)
    image = tf.reshape(image, in_net_size + [3])
    # decode of_stack
    of_min_max_arr = tf.decode_raw(features['of_min_max_raw'], tf.float32)
    of_stack_arr = []
    for n in range(of_stack_num):
        of_ext = ['_x', '_y']
        for xy in range(2):
            of_name = 'of_' + str(n) + of_ext[xy]
            of_jpg = tf.cast(tf.image.decode_jpeg(features[of_name]), tf.float32)
            of_min = of_min_max_arr[2 * (n + xy)]
            of_max = of_min_max_arr[2 * (n + xy) + 1]
            of_org = of_jpg * (of_max - of_min) / 255. + of_min
            if n + xy == 0:
                of_stack_arr = of_org
            else:
                of_stack_arr = tf.concat([of_stack_arr, of_org], 2)
    of_stack_arr = tf.image.resize_images(of_stack_arr, in_net_size, method=0)
    of_stack_arr = tf.reshape(of_stack_arr, in_net_size + [2 * of_stack_num])
    # set batch
    if for_eval is False:
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size
        image_batch, of_stack_batch, label_batch = tf.train.shuffle_batch([image, of_stack_arr, label],
                                                                          batch_size=batch_size, capacity=capacity,
                                                                          num_threads=cpu_count(),
                                                                          min_after_dequeue=min_after_dequeue)
        return image_batch, of_stack_batch, label_batch
    else:
        # decode frame info
        rootb = features['root']
        nameb = features['name']
        idexb = features['idex']
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size
        image_batch, of_stack_batch, label_batch, root_batch, name_batch, index_batch = tf.train.batch(
            [image, of_stack_arr, label, rootb, nameb, idexb],
            batch_size=batch_size, capacity=capacity,
            num_threads=cpu_count())
        return image_batch, of_stack_batch, label_batch, root_batch, name_batch, index_batch


def get_cifar10_input(cifar10_path):
    import pickle
    import os
    import numpy as np
    def dict_merge(a, b):
        for key in list(a.keys()):
            if type(a[key]) is list:
                a[key] = a[key] + b[key]
            elif type(a[key]) is np.ndarray:
                a[key] = np.row_stack((a[key], b[key]))
            elif type(a[key]) is bytes:
                a[key] = a[key] + b'ZBH' + b[key]
            else:
                exit('ZBH : wrong dict[key] type: %s' % key)
        return a

    test_file = os.path.join(cifar10_path, 'test_batch')
    with open(test_file, 'rb') as fo:
        test = pickle.load(fo, encoding='bytes')
    test[b'data'] = np.transpose(np.reshape(test[b'data'], [10000, 32, 32, 3], order='F'), (0, 2, 1, 3))
    train = {}
    for i in range(5):
        file = os.path.join(cifar10_path, 'data_batch_' + str(1 + i))
        with open(file, 'rb') as fo:
            temp = pickle.load(fo, encoding='bytes')
        temp[b'data'] = np.transpose(np.reshape(temp[b'data'], [10000, 32, 32, 3], order='F'), (0, 2, 1, 3))
        if not train:
            train = temp
        else:
            train = dict_merge(train, temp)
    return test, train
