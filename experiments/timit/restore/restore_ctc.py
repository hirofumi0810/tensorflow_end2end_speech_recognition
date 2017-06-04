#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Restore trained CTC network (TIMIT corpus)."""

import os
import sys
import tensorflow as tf
import yaml

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from data.read_dataset_ctc import DataSet
from models.ctc.load_model import load
from evaluation.eval_ctc import do_eval_per, do_eval_cer, decode_test, posterior_test


def do_restore(network, label_type, num_stack, num_skip, epoch=None):
    """Restore model.
    Args:
        network: model to restore
        label_type: phone39 or phone48 or phone61 or character
        num_stack: int, the number of frames to stack
        num_skip: int, the number of frames to skip
        epoch: epoch to restore
    """
    # Load dataset
    test_data = DataSet(data_type='test', label_type=label_type,
                        num_stack=num_stack, num_skip=num_skip,
                        is_sorted=False, is_progressbar=True)

    # Add to the graph each operation
    decode_op = network.decoder(decode_type='beam_search',
                                beam_width=20)
    posteriors_op = network.posteriors(decode_op)
    per_op = network.ler(decode_op)

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(network.model_dir)

        # If check point exists
        if ckpt:
            # Use last saved model
            model_path = ckpt.model_checkpoint_path
            if epoch is not None:
                model_path = model_path.split('/')[:-1]
                model_path = '/'.join(model_path) + '/model.ckpt-' + str(epoch)
            saver.restore(sess, model_path)
            print("Model restored: " + model_path)
        else:
            raise ValueError('There are not any checkpoints.')

        print('Test Data Evaluation:')
        if label_type == 'character':
            do_eval_cer(session=sess, decode_op=decode_op, network=network,
                        dataset=test_data, is_progressbar=True)
        else:
            do_eval_per(session=sess, decode_op=decode_op, per_op=per_op,
                        network=network, dataset=test_data,
                        label_type=label_type, is_progressbar=True)

        # Visualize
        decode_test(session=sess, decode_op=decode_op, network=network,
                    dataset=test_data, label_type=label_type)
        # posterior_test(session=sess, posteriors_op=posteriors_op,
        # network=network, dataset=test_data, label_type=label_type)


def main(model_path):

    epoch = None  # if None, restore the final epoch

    # Read config file
    with open(os.path.join(model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        corpus = config['corpus']
        feature = config['feature']
        param = config['param']

    if corpus['label_type'] == 'phone61':
        output_size = 61
    elif corpus['label_type'] == 'phone48':
        output_size = 48
    elif corpus['label_type'] == 'phone39':
        output_size = 39
    elif corpus['label_type'] == 'character':
        output_size = 30

    # Load model
    CTCModel = load(model_type=config['model_name'])
    network = CTCModel(batch_size=1,
                       input_size=feature['input_size'] * feature['num_stack'],
                       num_cell=param['num_cell'],
                       num_layers=param['num_layer'],
                       output_size=output_size,
                       clip_gradients=param['clip_grad'],
                       clip_activation=param['clip_activation'],
                       dropout_ratio_input=param['dropout_input'],
                       dropout_ratio_hidden=param['dropout_hidden'],
                       num_proj=param['num_proj'],
                       weight_decay=param['weight_decay'])
    network.model_name = config['model_name']
    network.model_dir = model_path

    print(network.model_dir)
    do_restore(network=network,
               label_type=corpus['label_type'],
               num_stack=feature['num_stack'],
               num_skip=feature['num_skip'],
               epoch=epoch)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 2:
        raise ValueError(("Set a path to saved model.\n"
                          "Usase: python restore_ctc.py path_to_saved_model"))

    main(model_path=args[1])
