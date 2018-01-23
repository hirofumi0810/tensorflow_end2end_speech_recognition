#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate trained CTC network (SVC corpus)."""

import os
import sys
import time
import tensorflow as tf

sys.path.append(os.path.pardir)
sys.path.append(os.path.abspath('../../../../'))
from feature_extraction.read_dataset_ctc import DataSet
from models.tf_model.ctc.load_model import load
from evaluation.eval_ctc import do_eval_ler, do_eval_fmeasure, do_eval_fmeasure_time


def do_restore(network, label_type, feature, epoch=None, is_progressbar=True):
    """Restore model.
    Args:
        network: model to restore
        label_type: original or phone1 or phone2 or phone41
        feature: fbank or is13
        epoch: the number of epoch to restore
        is_progressbar: if True, visualize progressbar
    """
    # load dataset
    print('Loading dataset...')
    test_data = DataSet(data_type='test', label_type=label_type,
                        feature=feature, is_progressbar=is_progressbar)

    # reset previous network
    tf.reset_default_graph()

    # add to the graph each operation
    network.define()
    # decode_op = network.greedy_decoder()
    decode_op = network.beam_search_decoder(beam_width=20)
    posteriors_op = network.posteriors(decode_op)
    ler_op = network.ler(decode_op)

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

        start_time_eval = time.time()

        print('■Test Data Evaluation:■')
        # do_eval_ler(session=sess, ler_op=ler_op, network=network,
        #             dataset=test_data, is_progressbar=is_progressbar)
        print('Fmeasure (time)')
        do_eval_fmeasure_time(session=sess, decode_op=decode_op, posteriors_op=posteriors_op,
                              network=network, dataset=test_data, is_progressbar=is_progressbar)

        print('F-measure (sequence)')
        do_eval_fmeasure(session=sess, decode_op=decode_op, network=network,
                         dataset=test_data, is_progressbar=is_progressbar)

        # sys.stdout.flush()
        duration_eval = time.time() - start_time_eval
        print('Evaluation time: %.3f min' % (duration_eval / 60))


def main():

    label_type = 'phone1'  # phone1 or phone2 or phone41
    feature = 'is13'  # fbank or is13
    model = 'blstm'
    layer_num = 5
    cell = 256
    optimizer = 'rmsprop'
    learning_rate = 1e-3
    epoch = 75
    drop_in = 0.8
    drop_h = 0.5
    save_result = True

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

    CTCModel = load(model_type=model)
    network = CTCModel(batch_size=32, input_size=input_size,
                       num_cell=cell, num_layers=layer_num,
                       output_size=output_size,
                       clip_grad=5.0, clip_activation=50,  # 推論時はどうする？
                       dropout_ratio_input=drop_in,
                       dropout_ratio_hidden=drop_h)
    network.model_name = model.upper() + '_CTC'
    network.model_name += '_' + str(cell) + '_' + str(layer_num)
    network.model_name += '_' + optimizer + '_lr' + str(learning_rate)

    # set restore path
    network.model_dir = '/n/sd8/inaguma/result/svc/ctc/'
    network.model_dir = os.path.join(network.model_dir, label_type)
    network.model_dir = os.path.join(network.model_dir, feature)
    network.model_dir = os.path.join(network.model_dir, network.model_name)

    if save_result:
        sys.stdout = open(os.path.join(network.model_dir, 'test_eval.txt'), 'w')
        print(network.model_dir)
        do_restore(network, label_type=label_type, feature=feature,
                   epoch=epoch, is_progressbar=False)
        sys.stdout = sys.__stdout__
    else:
        print(network.model_dir)
        do_restore(network, label_type=label_type, feature=feature,
                   epoch=epoch, is_progressbar=True)


if __name__ == '__main__':
    main()
