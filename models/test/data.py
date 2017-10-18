#! /usr/bin/env python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.io.inputs.splicing import do_splice
from utils.io.inputs.feature_extraction import wav2feature
from utils.io.labels.phone import Phone2idx

SPACE = '_'
SOS = '<'
EOS = '>'


def _read_text(text_path):
    """Read char-level transcripts.
    Args:
        text_path (string): path to a transcript text file
    Returns:
        transcript (string): a text of transcript
    """
    # Read ground truth labels
    with open(text_path, 'r') as f:
        line = f.readlines()[-1]
        transcript = SPACE.join(line.strip().lower().split(' ')[2:])
    return transcript


def _read_phone(text_path):
    """Read phone-level transcripts.
    Args:
        text_path (string): path to a transcript text file
    Returns:
        transcript (string): a text of transcript
    """
    # Read ground truth labels
    phone_list = []
    with open(text_path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            phone_list.append(line[-1])
    transcript = ' '.join(phone_list)
    return transcript


def generate_data(label_type, model, batch_size=1, splice=1):
    """
    Args:
        label_type (string): character or phone or multitask
        model (string): ctc or attention or joint_ctc_attention
        batch_size (int, optional): the size of mini-batch
        splice (int, optional): frames to splice. Default is 1 frame.
    Returns:
        inputs: `[B, T, input_size]`
        labels: `[B]`
        inputs_seq_len: `[B, frame_num]`
        labels_seq_len: `[B]` (if model is attention)
    """
    # Make input data
    inputs, inputs_seq_len = wav2feature(
        ['./sample/LDC93S1.wav'] * batch_size,
        feature_type='logfbank', feature_dim=40,
        energy=False, delta1=True, delta2=True)

    # Splice
    inputs = do_splice(inputs, splice=splice)

    phone2idx = Phone2idx(map_file_path='./phone61.txt')

    trans_char = _read_text('./sample/LDC93S1.txt')
    trans_char = trans_char.replace('.', '')
    trans_phone = _read_phone('./sample/LDC93S1.phn')

    # Make transcripts
    if model == 'ctc':
        if label_type == 'character':
            labels = [alpha2idx(trans_char)] * batch_size
            return inputs, labels, inputs_seq_len

        elif label_type == 'phone':
            labels = [phone2idx(trans_phone.split(' '))] * batch_size
            return inputs, labels, inputs_seq_len

        elif label_type == 'multitask':
            labels_char = [alpha2idx(trans_char)] * batch_size
            labels_phone = [phone2idx(trans_phone.split(' '))] * batch_size
            return inputs, labels_char, labels_phone, inputs_seq_len

    elif model == 'attention':
        if label_type == 'character':
            trans_char = SOS + trans_char + EOS
            labels = [alpha2idx(trans_char)] * batch_size
            labels_seq_len = [len(labels[0])] * batch_size
            return inputs, labels, inputs_seq_len, labels_seq_len

        elif label_type == 'phone':
            trans_phone = SOS + ' ' + trans_phone + ' ' + EOS
            labels = [phone2idx(trans_phone.split(' '))] * batch_size
            labels_seq_len = [len(labels[0])] * batch_size
            return inputs, labels, inputs_seq_len, labels_seq_len

        elif label_type == 'multitask':
            trans_char = SOS + trans_char + EOS
            trans_phone = SOS + ' ' + trans_phone + ' ' + EOS
            labels_char = [alpha2idx(trans_char)] * batch_size
            labels_phone = [phone2idx(trans_phone.split(' '))] * batch_size
            target_len_char = [len(labels_char[0])] * batch_size
            target_len_phone = [len(labels_phone[0])] * batch_size
            return (inputs, labels_char, labels_phone,
                    inputs_seq_len, target_len_char, target_len_phone)

    elif model == 'joint_ctc_attention':
        if label_type == 'character':
            att_trans_char = SOS + trans_char + EOS
            att_labels = [alpha2idx(att_trans_char)] * batch_size
            labels_seq_len = [len(att_labels[0])] * batch_size
            ctc_labels = [alpha2idx(trans_char)] * batch_size
        elif label_type == 'phone':
            att_trans_phone = SOS + ' ' + trans_phone + ' ' + EOS
            att_labels = [phone2idx(att_trans_phone.split(' '))] * batch_size
            labels_seq_len = [len(att_labels[0])] * batch_size
            ctc_labels = [phone2idx(trans_phone.split(' '))] * batch_size
        return inputs, att_labels, ctc_labels, inputs_seq_len, labels_seq_len


def alpha2idx(transcript):
    """Convert from alphabet to number.
    Args:
        transcript (string): a sequence of characters
    Returns:
        index_list (list): indices of alphabets
    """
    char_list = list(transcript)

    first_index = ord('a')
    index_list = []
    for char in char_list:
        if char == SPACE:
            index_list.append(26)
        elif char == SOS:
            index_list.append(27)
        elif char == EOS:
            index_list.append(28)
        else:
            index_list.append(ord(char) - first_index)
    return index_list


def idx2alpha(index_list):
    """Convert from number to alphabet.
    Args:
        index_list (list): indices of alphabets
    Returns:
        transcript (string): a sequence of characters
    """
    first_index = ord('a')
    char_list = []
    for num in index_list:
        if num == 26:
            char_list.append(SPACE)
        elif num == 27:
            char_list.append(SOS)
        elif num == 28:
            char_list.append(EOS)
        else:
            char_list.append(chr(num + first_index))
    transcript = ''.join(char_list)
    return transcript
