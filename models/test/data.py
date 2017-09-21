#! /usr/bin/env python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.data.sparsetensor import list2sparsetensor
from utils.data.labels.phone import phone2idx
from utils.data.inputs.splicing import do_splice
from input_pipeline.feature_extraction import wav2feature


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
        transcript = ' '.join(line.strip().lower().split(' ')[2:])
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
        energy=True, delta1=True, delta2=True)

    # Splice
    inputs = do_splice(inputs, splice=splice)

    ctc_phone_map_file_path = '../../experiments/timit/metrics/mapping_files/ctc/phone61.txt'
    att_phone_map_file_path = '../../experiments/timit/metrics/mapping_files/attention/phone61.txt'

    # Make transcripts
    if model == 'ctc':
        if label_type == 'character':
            transcript = _read_text('./sample/LDC93S1.txt')
            transcript = ' ' + transcript.replace('.', '') + ' '
            labels = [alpha2idx(transcript)] * batch_size

            # Convert to SparseTensor
            labels = list2sparsetensor(labels, padded_value=-1)
            return inputs, labels, inputs_seq_len

        elif label_type == 'phone':
            transcript = _read_phone('./sample/LDC93S1.phn')
            labels = [
                phone2idx(transcript.split(' '), ctc_phone_map_file_path)] * batch_size

            # Convert to SparseTensor
            labels = list2sparsetensor(labels, padded_value=-1)
            return inputs, labels, inputs_seq_len

        elif label_type == 'multitask':
            transcript_char = _read_text('./sample/LDC93S1.txt')
            transcript_phone = _read_phone('./sample/LDC93S1.phn')
            transcript_char = ' ' + transcript_char.replace('.', '') + ' '
            labels_char = [alpha2idx(transcript_char)] * batch_size
            labels_phone = [
                phone2idx(transcript_phone.split(' '), ctc_phone_map_file_path)] * batch_size

            # Convert to SparseTensor
            labels_char = list2sparsetensor(labels_char, padded_value=-1)
            labels_phone = list2sparsetensor(labels_phone, padded_value=-1)
            return inputs, labels_char, labels_phone, inputs_seq_len

    elif model == 'attention':
        if label_type == 'character':
            transcript = _read_text('./sample/LDC93S1.txt')
            transcript = '<' + transcript.replace('.', '') + '>'
            labels = [alpha2idx(transcript)] * batch_size
            labels_seq_len = [len(labels[0])] * batch_size
            return inputs, labels, inputs_seq_len, labels_seq_len

        elif label_type == 'phone':
            transcript = _read_phone('./sample/LDC93S1.phn')
            transcript = '< ' + transcript + ' >'
            labels = [phone2idx(transcript.split(
                ' '), att_phone_map_file_path)] * batch_size
            labels_seq_len = [len(labels[0])] * batch_size
            return inputs, labels, inputs_seq_len, labels_seq_len

        elif label_type == 'multitask':
            transcript_char = _read_text('./sample/LDC93S1.txt')
            transcript_phone = _read_phone('./sample/LDC93S1.phn')
            transcript_char = '<' + transcript_char.replace('.', '') + '>'
            transcript_phone = '< ' + transcript_phone + ' >'
            labels_char = [alpha2idx(transcript_char)] * batch_size
            labels_phone = [
                phone2idx(transcript_phone.split(' '), att_phone_map_file_path)] * batch_size
            target_len_char = [len(labels_char[0])] * batch_size
            target_len_phone = [len(labels_phone[0])] * batch_size
            return (inputs, labels_char, labels_phone,
                    inputs_seq_len, target_len_char, target_len_phone)

    elif model == 'joint_ctc_attention':
        if label_type == 'character':
            transcript = _read_text('./sample/LDC93S1.txt')
            att_transcript = '<' + transcript.replace('.', '') + '>'
            ctc_transcript = ' ' + transcript.replace('.', '') + ' '
            att_labels = [alpha2idx(att_transcript)] * batch_size
            labels_seq_len = [len(att_labels[0])] * batch_size
            ctc_labels = [alpha2idx(ctc_transcript)] * batch_size

            # Convert to SparseTensor
            ctc_labels = list2sparsetensor(ctc_labels, padded_value=-1)
            return inputs, att_labels, inputs_seq_len, labels_seq_len, ctc_labels

        elif label_type == 'phone':
            transcript = _read_phone('./sample/LDC93S1.phn')
            att_transcript = '< ' + transcript + ' >'
            att_labels = [
                phone2idx(att_transcript.split(' '), att_phone_map_file_path)] * batch_size
            labels_seq_len = [len(att_labels[0])] * batch_size
            ctc_labels = [
                phone2idx(transcript.split(' '), ctc_phone_map_file_path)] * batch_size

            # Convert to SparseTensor
            ctc_labels = list2sparsetensor(ctc_labels, padded_value=-1)

            return inputs, att_labels, inputs_seq_len, labels_seq_len, ctc_labels


def alpha2idx(transcript):
    """Convert from alphabet to number.
    Args:
        transcript (string): a sequence of characters
    Returns:
        index_list (list): indices of alphabets
    """
    char_list = list(transcript)

    # 0 is reserved for space
    space_index = 0
    first_index = ord('a') - 1
    index_list = []
    for char in char_list:
        if char == ' ':
            index_list.append(space_index)
        elif char == '<':
            index_list.append(26)
        elif char == '>':
            index_list.append(27)
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
    # 0 is reserved to space
    first_index = ord('a') - 1
    char_list = []
    for num in index_list:
        if num == 0:
            char_list.append(' ')
        elif num == 26:
            char_list.append('<')
        elif num == 27:
            char_list.append('>')
        else:
            char_list.append(chr(num + first_index))
    transcript = ''.join(char_list)
    return transcript
