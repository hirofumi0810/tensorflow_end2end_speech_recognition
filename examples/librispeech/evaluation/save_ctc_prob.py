#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Save the trained CTC posteriors (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import numpy as np
import tensorflow as tf
import yaml
import argparse
from tqdm import tqdm
import random

sys.path.append(abspath('../../../'))
from experiments.librispeech.data.load_dataset_ctc import Dataset
from models.ctc.ctc import CTC
from utils.directory import mkdir_join
from utils.io.inputs.splicing import do_splice
from utils.parallel import make_parallel


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--eval_batch_size', type=int, default=-1,
                    help='the size of mini-batch when evaluation. ' +
                    'If you set -1, batch size is the same as that when training.')
parser.add_argument('--temperature', type=int, default=1,
                    help='temperature parameter')

TOTAL_NUM_FRAMES_DICT = {
    "train": 18088388,
    "dev_clean": 968057,
    "dev_other": 919980,
    "test_clean": 970953,
    "test_other": 959587
}

NUM_POOLS = 10


def do_save(model, params, epoch, eval_batch_size, temperature):
    """Save the CTC outputs.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
        eval_batch_size (int): the size of mini-batch in evaluation
        temperature (int):
    """
    print('=' * 30)
    print('  frame stack %d' % int(params['num_stack']))
    print('  splice %d' % int(params['splice']))
    print('  temperature (training): %d' % temperature)
    print('=' * 30)

    # Load dataset
    train_data = Dataset(
        data_type='train', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'] if eval_batch_size == -
        1 else eval_batch_size,
        max_epoch=3, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=True, num_gpu=1)
    dev_clean_data = Dataset(
        data_type='dev_clean', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'] if eval_batch_size == -
        1 else eval_batch_size,
        max_epoch=3, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=True, num_gpu=1)
    dev_other_data = Dataset(
        data_type='dev_other', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'] if eval_batch_size == -
        1 else eval_batch_size,
        max_epoch=3, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=True, num_gpu=1)
    test_clean_data = Dataset(
        data_type='test_clean', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'] if eval_batch_size == -
        1 else eval_batch_size,
        max_epoch=3, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=True, num_gpu=1)
    test_other_data = Dataset(
        data_type='test_other', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'] if eval_batch_size == -
        1 else eval_batch_size,
        max_epoch=3, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=True, num_gpu=1)

    with tf.name_scope('tower_gpu0'):
        # Define placeholders
        model.create_placeholders()

        # Add to the graph each operation (including model definition)
        _, logits = model.compute_loss(
            model.inputs_pl_list[0],
            model.labels_pl_list[0],
            model.inputs_seq_len_pl_list[0],
            model.keep_prob_pl_list[0])
        logits /= temperature
        posteriors_op = model.posteriors(logits, blank_prior=1)

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model.save_path)

        # If check point exists
        if ckpt:
            model_path = ckpt.model_checkpoint_path
            if epoch != -1:
                model_path = model_path.split('/')[:-1]
                model_path = '/'.join(model_path) + '/model.ckpt-' + str(epoch)
            saver.restore(sess, model_path)
            print("Model restored: " + model_path)
        else:
            raise ValueError('There are not any checkpoints.')

        #########################
        # Save soft targets
        #########################
        # train100h
        # save(session=sess,
        #      posteriors_op=posteriors_op,
        #      model=model,
        #      dataset=train_data,
        #      data_type='train',
        #      num_stack=params['num_stack'],
        #      save_prob=False,
        #      save_soft_targets=True,
        #      save_path=mkdir_join(model.save_path, 'temp' + str(temperature), 'train'))

        # dev
        # save(session=sess,
        #      posteriors_op=posteriors_op,
        #      model=model,
        #      dataset=dev_clean_data,
        #      data_type='dev_clean',
        #      num_stack=params['num_stack'],
        #      save_prob=False,
        #      save_soft_targets=True,
        #      save_path=mkdir_join(model.save_path, 'temp' + str(temperature), 'dev_clean'))
        # save(session=sess,
        #      posteriors_op=posteriors_op,
        #      model=model,
        #      dataset=dev_other_data,
        #      data_type='dev_other',
        #      num_stack=params['num_stack'],
        #      save_prob=False,
        #      save_soft_targets=True,
        #      save_path=mkdir_join(model.save_path, 'temp' + str(temperature), 'dev_other'))

        # test
        save(session=sess,
             posteriors_op=posteriors_op,
             model=model,
             dataset=test_clean_data,
             data_type='test_clean',
             num_stack=params['num_stack'],
             save_prob=True,
             save_soft_targets=False,
             save_path=mkdir_join(model.save_path, 'temp' + str(temperature), 'test_clean'))
        save(session=sess,
             posteriors_op=posteriors_op,
             model=model,
             dataset=test_other_data,
             data_type='test_other',
             num_stack=params['num_stack'],
             save_prob=True,
             save_soft_targets=False,
             save_path=mkdir_join(model.save_path, 'temp' + str(temperature), 'test_other'))


def save(session, posteriors_op, model, dataset, data_type,
         save_prob=False, save_soft_targets=False,
         num_stack=1, save_path=None):

    # Initialize
    pbar = tqdm(total=len(dataset))
    total_num_frames = 0
    pool_input_frames = None
    pool_prob_frames = None
    num_frames_per_block = 1024 * 100
    frame_counter = 0
    block_counter = 0
    pool_counter = 0
    accumulated_total_num_frames = 0

    ########################################
    # Count total frame number
    ########################################
    # for data, is_new_epoch in dataset:
    #
    #     # Create feed dictionary for next mini batch
    #     inputs, _, inputs_seq_len, input_names = data
    #
    #     batch_size = inputs[0].shape[0]
    #     for i_batch in range(batch_size):
    #         total_num_frames += inputs_seq_len[0][i_batch]
    #
    #         pbar.update(1)
    #
    #     if is_new_epoch:
    #         print(total_num_frames)
    #         break

    ########################################
    # Save probabilities per utterance
    ########################################
    pbar = tqdm(total=len(dataset))
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, _, inputs_seq_len, input_names = data
        feed_dict = {
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            model.keep_prob_pl_list[0]: 1.0
        }

        batch_size, max_time = inputs[0].shape[:2]

        probs = session.run(posteriors_op, feed_dict=feed_dict)
        probs = probs.reshape(batch_size, max_time, model.num_classes)

        if pool_input_frames is None:
            # Initialize
            total_num_frames = TOTAL_NUM_FRAMES_DICT[data_type]

            pool_num_frames = total_num_frames // NUM_POOLS + 1
            pool_capacity = pool_num_frames

            pool_input_frames = np.zeros(
                (pool_num_frames, 120 * 2 * 5))
            # NOTE: input_size == 120 * 2 (num_stack == 2), splice == 5
            pool_prob_frames = np.zeros(
                (pool_num_frames, model.num_classes))

        for i_batch in range(batch_size):
            speaker = input_names[0][i_batch].split('-')[0]

            # Mask
            inputs_seq_len_i = inputs_seq_len[0][i_batch]
            inputs_i = inputs[0][i_batch][:inputs_seq_len_i]
            probs_i = probs[i_batch][:inputs_seq_len_i]

            # Save probabilities as npy file per utterance
            if save_prob:
                prob_save_path = mkdir_join(
                    save_path, 'probs_utt', speaker, input_names[0][i_batch] + '.npy')
                np.save(prob_save_path, probs_i)
                # NOTE: `[T, num_classes]`

            if dataset.splice == 1:
                # NOTE: teacher is expected to be BLSTM
                # Splicing
                inputs_i = do_splice(inputs_i.reshape(1, inputs_seq_len_i, -1),
                                     splice=5,
                                     batch_size=1,
                                     num_stack=dataset.num_stack)
                inputs_i = inputs_i.reshape(inputs_seq_len_i, -1)

            else:
                # NOTE: teahcer is expected to be VGG (use features as it is)
                pass

            # Register
            if pool_capacity > inputs_seq_len_i:
                pool_input_frames[frame_counter:frame_counter +
                                  inputs_seq_len_i] = inputs_i
                pool_prob_frames[frame_counter: frame_counter +
                                 inputs_seq_len_i] = probs_i
                frame_counter += inputs_seq_len_i
                pool_capacity -= inputs_seq_len_i
            else:
                # Fulfill pool
                pool_input_frames[frame_counter:frame_counter +
                                  pool_capacity] = inputs_i[:pool_capacity]
                pool_prob_frames[frame_counter:frame_counter +
                                 pool_capacity] = probs_i[:pool_capacity]

                ##################################################
                # Shuffle frames, divide into blocks, and save
                ##################################################
                num_blocks = pool_num_frames // num_frames_per_block
                data_indices = list(range(pool_num_frames))
                random.shuffle(data_indices)

                for i_block in range(num_blocks):
                    block_indices = data_indices[:num_frames_per_block]
                    data_indices = data_indices[num_frames_per_block:]

                    # Pick up block
                    block_inputs_frames = pool_input_frames[block_indices]
                    # NOTE: `[1024 * 100, input_size]`
                    block_probs_frames = pool_prob_frames[block_indices]
                    # NOTE：`[1024 * 100, num_classes]`

                    # Save block
                    if save_soft_targets:
                        print(' ==> Saving: block%d' % block_counter)
                        input_save_path = mkdir_join(
                            save_path, 'inputs', 'block' + str(block_counter) + '.npy')
                        label_save_path = mkdir_join(
                            save_path, 'labels', 'block' + str(block_counter) + '.npy')
                        np.save(input_save_path, block_inputs_frames)
                        np.save(label_save_path, block_probs_frames)

                    block_counter += 1
                    accumulated_total_num_frames += len(block_indices)

                pool_carry_over_num_frames = pool_num_frames - num_frames_per_block * num_blocks
                utt_carry_over_num_frames = inputs_seq_len_i - pool_capacity
                carry_over_num_frames = pool_carry_over_num_frames + utt_carry_over_num_frames

                pool_carry_over_input_frames = pool_input_frames[data_indices]
                pool_carry_over_prob_frames = pool_prob_frames[data_indices]

                # Initialize
                if pool_counter != NUM_POOLS - 1:
                    pool_num_frames = total_num_frames // NUM_POOLS + 1 + carry_over_num_frames
                else:
                    # last pool
                    pool_num_frames = total_num_frames - accumulated_total_num_frames

                pool_input_frames = np.zeros(
                    (pool_num_frames, 120 * 2 * 5))
                # NOTE: input_size == 120 * 2 (num_stack == 2), splice == 5
                pool_prob_frames = np.zeros(
                    (pool_num_frames, model.num_classes))
                frame_counter = 0
                pool_counter += 1

                # Register carry over frames
                pool_input_frames[:pool_carry_over_num_frames] = pool_carry_over_input_frames
                pool_prob_frames[:pool_carry_over_num_frames] = pool_carry_over_prob_frames
                frame_counter += pool_carry_over_num_frames
                pool_input_frames[frame_counter:frame_counter +
                                  utt_carry_over_num_frames] = inputs_i[-utt_carry_over_num_frames:]
                pool_prob_frames[frame_counter:frame_counter +
                                 utt_carry_over_num_frames] = probs_i[-utt_carry_over_num_frames:]
                frame_counter += utt_carry_over_num_frames

                pool_capacity = pool_num_frames - carry_over_num_frames
                print('=== next pool ===')

        pbar.update(batch_size)

        if is_new_epoch:
            ##################################################
            # Save last pool
            ##################################################
            # Pick up block
            block_inputs_frames = pool_input_frames[:frame_counter]
            # NOTE: `[1024 * 100, input_size]`
            block_probs_frames = pool_prob_frames[:frame_counter]
            # NOTE：`[1024 * 100, num_classes]`

            # Save last lock
            if save_soft_targets:
                print(' ==> Saving: block%d' % block_counter)
                np.save(mkdir_join(save_path, 'inputs', 'block' +
                                   str(block_counter) + '.npy'), block_inputs_frames)
                np.save(mkdir_join(save_path, 'labels', 'block' +
                                   str(block_counter) + '.npy'), block_probs_frames)

            break


def main():

    args = parser.parse_args()

    # Load config file
    with open(join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank class
    if params['label_type'] == 'character':
        params['num_classes'] = 28

    # Model setting
    model = CTC(
        encoder_type=params['encoder_type'],
        input_size=params['input_size'] * params['num_stack'],
        splice=params['splice'],
        num_units=params['num_units'],
        num_layers=params['num_layers'],
        num_classes=params['num_classes'],
        lstm_impl=params['lstm_impl'],
        use_peephole=params['use_peephole'],
        parameter_init=params['weight_init'],
        clip_grad_norm=params['clip_grad_norm'],
        clip_activation=params['clip_activation'],
        num_proj=params['num_proj'],
        weight_decay=params['weight_decay'])

    model.save_path = args.model_path
    do_save(model=model, params=params, epoch=args.epoch,
            eval_batch_size=args.eval_batch_size,
            temperature=args.temperature)


if __name__ == '__main__':
    main()
