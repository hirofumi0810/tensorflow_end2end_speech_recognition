#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Restore trained feed-forward network (SVC corpus)."""

import os
import sys
import tensorflow as tf

sys.path.append(os.path.pardir)
sys.path.append(os.path.abspath('../../../../'))
from feature_extraction.read_dataset_ff import DataSet
from models.tf_model.feed_forward.dnn import dnnModel
from models.tf_model.feed_forward.cnn2_fc2 import cnnModel as cnn2
from evaluation.eval_framewise import do_eval_uaauc


def do_restore(network, label_type, feature, epoch=None):
    """Restore model.
    Args:
        network: model to restore
        label_type: original or phone2 or phone41
        epoch: the number of epoch to restore
        feature: fbank or is13
    """
    # load dataset
    print('Loading dataset...')
    test_data = DataSet(data_type='test', feature=feature,
                        label_type=label_type, is_progressbar=True)

    # add to the graph each operation
    network.define()
    posteriors_op = network.posterior()

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

        print('Test Data Evaluation:')
        do_eval_uaauc(session=sess, posteriors_op=posteriors_op,
                      network=network, dataset=test_data, is_training=False)


def main():

    label_type = 'original'  # original or phone2 or phone41
    feature = 'fbank'  # fbank or is13
    model = 'dnn'
    node_list = [400] * 4
    optimizer = 'adam'
    learning_rate = 1e-3
    epoch = 1
    dropout = True
    batch_norm = True

    if feature == 'fbank':
        input_size = 123
    elif feature == 'is13':
        input_size = 141

    if label_type == 'original':
        output_size = 3
    elif label_type == 'phone2':
        output_size = 4
    elif label_type == 'phone41':
        output_size = 43

    if model == 'dnn':
        network = dnnModel(batch_size=1, input_size=input_size, splice=5,
                           hidden_size_list=node_list, output_size=output_size,
                           is_dropout=dropout, dropout_ratio_list=[0.5] * len(node_list),
                           is_batch_norm=batch_norm)
        network.model_name = 'DNN'
    elif model == 'cnn2':
        pass
        # network = cnn2(batch_size=1, input_size=141, splice=SPLICE, output_size=3,
        #                weight_decay_lambda=0.0)
    else:
        raise ValueError('Error: model is "dnn" or "cnn2".')

    network.model_name += '_' + str(node_list[0]) + '_'
    network.model_name += str(len(node_list)) + '_' + optimizer
    network.model_name += '_lr' + str(learning_rate)

    network.model_dir = '/n/sd8/inaguma/result/svc/framewise/'
    network.model_dir = os.path.join(network.model_dir, label_type)
    network.model_dir = os.path.join(network.model_dir, feature)
    network.model_dir = os.path.join(network.model_dir, network.model_name)

    print(network.model_dir)
    do_restore(network, label_type=label_type, feature=feature, epoch=epoch)


if __name__ == '__main__':
    main()
