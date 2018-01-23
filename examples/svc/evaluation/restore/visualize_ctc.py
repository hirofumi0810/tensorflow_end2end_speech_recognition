#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Visualize trained CTC network (SVC corpus)."""

import os
import sys
import tensorflow as tf

sys.path.append(os.path.pardir)
sys.path.append(os.path.abspath('../../../../'))
from feature_extraction.read_dataset_ctc import DataSet
from models.tf_model.ctc.load_model import load
from evaluation.eval_ctc import decode_test, posterior_test


def do_restore(network, label_type, feature, epoch=None):
    """Restore model.
    Args:
        network: model to restore
        label_type: original or phone1 or phone2 or phone41
        epoch: the number of epoch to restore
        feature: fbank or is13
    """
    # load dataset
    print('Loading dataset...')
    test_data = DataSet(data_type='test', feature=feature,
                        label_type=label_type, is_progressbar=True)

    # reset previous network
    tf.reset_default_graph()

    # add to the graph each operation
    network.define()
    # decode_op = network.greedy_decoder()
    decode_op = network.beam_search_decoder(beam_width=20)
    posteriors_op = network.posteriors(decode_op)

    # create a saver for writing training checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(network.model_dir)

        # if check point exists
        if ckpt:
            # use last saved model
            model_path = ckpt.model_checkpoint_path
            if epoch is not None:
                model_path = model_path.split('/')[:-1]
                model_path = '/'.join(model_path) + '/model.ckpt-' + str(epoch)
            saver.restore(sess, model_path)
            print("Model restored: " + model_path)
        else:
            raise ValueError('There are not any checkpoints.')

        # visualize
        # decode_test(session=sess, decode_op=decode_op,
        #             network=network, dataset=test_data, label_type=label_type)
        posterior_test(session=sess, posteriors_op=posteriors_op,
                       network=network, dataset=test_data, label_type=label_type)


def main():

    label_type = 'phone1'  # phone1 or phone2 or phone41
    feature = 'fbank'  # fbank or is13
    model = 'blstm'
    layer_num = 5
    cell = 256
    optimizer = 'rmsprop'
    learning_rate = 1e-3
    epoch = 92
    input_drop = 0.8
    hidden_drop = 0.5

    if feature == 'fbank':
        input_size = 123
    elif feature == 'is13':
        input_size = 141

    if label_type in ['original', 'phone1']:
        output_size = 3
    elif label_type == 'phone2':
        output_size = 4
    elif label_type == 'phone41':
        output_size = 43

    # load model
    CTCModel = load(model_type=model)
    network = CTCModel(batch_size=32, input_size=input_size,
                       num_cell=cell, num_layers=layer_num,
                       output_size=output_size,
                       clip_grad=5.0, clip_activation=50,  # 推論時はどうする？
                       dropout_ratio_input=input_drop,
                       dropout_ratio_hidden=hidden_drop)
    network.model_name = model.upper() + '_CTC'
    network.model_name += '_' + str(cell) + '_' + str(layer_num)
    network.model_name += '_' + optimizer + '_lr' + str(learning_rate)

    # set restore path
    network.model_dir = '/n/sd8/inaguma/result/svc/ctc/'
    network.model_dir = os.path.join(network.model_dir, label_type)
    network.model_dir = os.path.join(network.model_dir, feature)
    network.model_dir = os.path.join(network.model_dir, network.model_name)

    print(network.model_dir)
    do_restore(network, label_type=label_type, feature=feature, epoch=epoch)


if __name__ == '__main__':
    main()
