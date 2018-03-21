import time
import tensorflow as tf
import alex2_inference as net_inference
import z_train_cifar10 as net_train
import datasets_reader as dr
import os

EVAL_INTERVAL_SECS = 10


def evaluate():
    # dr.check_or_create_path([net_train.DATA_TF_FILE_PATH])
    [class_list_init, train_list_init, test_list_init] = dr.get_train_and_test_list(net_train.KEY,
                                                                                    net_train.FILE_LIST_PATH,
                                                                                    net_train.split)
    [class_list, _, test_list] = dr.if_make_demo_list(class_list_init, train_list_init, test_list_init,
                                                      net_train.DEMO)
    # dr.naive_check_multiprocessing(class_list, net_train.DATA_FRAMES_PATH, net_train.DATA_OF_PATH,
    #                               net_train.DATA_OF_DIC_PATH)
    # dr.check_or_create_tfrecords_multiprocessing(test_list, class_list, net_train.DATA_FRAMES_PATH,
    #                                              net_train.DATA_TF_FILE_PATH, net_train.OF_STACK_NUMS, )
    tfrecord_list = dr.create_tfrecords_list(test_list, net_train.DATA_TF_FILE_PATH)



    with tf.Graph().as_default() as g:
        image_batch, of_stack_batch, label_batch, root_batch, name_batch, index_batch = net_train.get_2stream_input(
            tfrecord_list, for_eval=True)
        with tf.variable_scope(tf.get_variable_scope()):
            label_batch_ = net_inference.inference(image_batch, 3, len(class_list), None, False, net_stream='spatial')
        variable_averages = tf.train.ExponentialMovingAverage(net_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        os.environ["CUDA_VISIBLE_DEVICES"] = '3'
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        while True:
            with tf.Session(config=config) as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)


                # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                ckpt = tf.train.get_checkpoint_state(net_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    whole = []
                    label_all_ground = []
                    label_all_eval = []
                    batch_go_on = True
                    while batch_go_on is True:
                        aa, bb, cc, dd, ee = sess.run([root_batch, name_batch, index_batch,
                                                       tf.argmax(label_batch_, 1), label_batch])
                        temp = list(zip(aa, bb, cc))
                        whole = whole + temp

                        label_all_ground = label_all_ground + list(ee)
                        label_all_eval = label_all_eval + list(dd)

                        if whole[0] in whole[-net_train.BATCH_SIZE:] and len(whole) > net_train.BATCH_SIZE:
                            whole_end = whole.index(whole[0], -net_train.BATCH_SIZE)
                            whole = whole[:whole_end]
                            label_all_ground = label_all_ground[:whole_end]
                            label_all_eval = label_all_eval[:whole_end]
                            batch_go_on = False
                        print('*****************')
                        print(dd)
                        print(ee)
                    print("After %s training step(s), validation accuracy = 100" % (global_step))
                else:
                    print('No checkpoint file found')
                    return

                coord.request_stop()
                coord.join(threads)

                time.sleep(EVAL_INTERVAL_SECS)


def main():
    evaluate()


if __name__ == '__main__':
    main()
