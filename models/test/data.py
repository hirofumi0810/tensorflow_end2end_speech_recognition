#! /usr/bin/env python
# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io.wavfile
from python_speech_features import mfcc, fbank, logfbank, hz2mel


def read_wav(wav_path, feature_type='logmelfbank', batch_size=1):
    """Read wav file & convert to MFCC or log mel filterbank features.
    Args:
        wav_path: path to a wav file
        feature: logmelfbank or mfcc
    Returns:
        inputs: `[batch_size, max_time, feature_dim]`
        seq_len: `[batch_size, frame_num]`
    """
    # Load wav file
    fs, audio = scipy.io.wavfile.read(wav_path)

    if feature_type == 'mfcc':
        features = mfcc(audio, samplerate=fs)  # (291, 13)
    elif feature_type == 'logmelfbank':
        fbank_features, energy = fbank(audio, nfilt=40)
        logfbank = np.log(fbank_features)
        logenergy = np.log(energy)
        logmelfbank = hz2mel(logfbank)
        features = np.c_[logmelfbank, logenergy]  # (291, 41)

    delta1 = delta(features, N=2)
    delta2 = delta(delta1, N=2)
    input_data = np.c_[features, delta1, delta2]  # (291, 123)

    # Transform to 3D array
    # (1, 291, 39) or (1, 291, 123)
    inputs = np.zeros((batch_size, input_data.shape[0], input_data.shape[1]))
    for i in range(batch_size):
        inputs[i] = input_data
    seq_len = [inputs.shape[1]] * batch_size  # [291]

    # Normalization
    inputs = (inputs - np.mean(inputs)) / np.std(inputs)

    return inputs, seq_len


def delta(feat, N):
    """Compute delta features from a feature vector sequence.
    Args:
        feat: A numpy array of size (NUMFRAMES by number of features) containing features.
              Each row holds 1 feature vector.
        N: For each frame, calculate delta features based on preceding and following N frames.
    Rreturns:
        dfeat: A numpy array of size (NUMFRAMES by number of features) containing delta features.
               Each row holds 1 delta feature vector.
    """
    NUMFRAMES = len(feat)
    feat = np.concatenate(([feat[0] for i in range(N)],
                           feat, [feat[-1] for i in range(N)]))
    denom = sum([2 * i * i for i in range(1, N + 1)])
    dfeat = []
    for j in range(NUMFRAMES):
        dfeat.append(np.sum([n * feat[N + j + n]
                             for n in range(-1 * N, N + 1)], axis=0) / denom)
    return dfeat


def read_text(text_path):
    """Read char-level transcripts.
    Args:
        text_path: path to a transcript text file
    Returns:
        transcript: a text of transcript
    """
    # Read ground truth labels
    with open(text_path, 'r') as f:
        line = f.readlines()[-1]
        transcript = ' '.join(line.strip().lower().split(' ')[2:])
    return transcript


def read_phone(text_path):
    """Read phone-level transcripts.
    Args:
        text_path: path to a transcript text file
    Returns:
        transcript: a text of transcript
    """
    # Read ground truth labels
    phone_list = []
    with open(text_path, 'r') as f:
        for line in f:
            line = line.strip().split(' ')
            phone_list.append(line[-1])
    transcript = ' '.join(phone_list)
    return transcript


def generate_data(label_type, model, batch_size=1):
    """
    Args:
        label_type: character or phone or multitask
        model: ctc or attention
    Returns:
        inputs: `[batch_size, max_time, feature_dim]`
        labels:
        seq_len: `[batch_size, frame_num]`
    """
    # Make input data
    inputs, seq_len = read_wav('./sample/LDC93S1.wav',
                               feature_type='logmelfbank',
                               batch_size=batch_size)
    # inputs, seq_len = read_wav('../sample/LDC93S1.wav',
    #                            feature_type='mfcc',
    #                            batch_size=batch_size)

    if label_type == 'character':
        transcript = read_text('./sample/LDC93S1.txt')
        if model == 'ctc':
            transcript = ' ' + transcript.replace('.', '') + ' '
            labels = [alpha2num(transcript)]
        elif model == 'attention':
            transcript = '<' + transcript.replace('.', '') + '>'
            labels = [alpha2num(transcript)]
    elif label_type == 'phone':
        transcript = read_phone('./sample/LDC93S1.phn')
        if model == 'ctc':
            labels = [phone2num(transcript)]
        elif model == 'attention':
            labels = ['<'] + [phone2num(transcript)] + ['>']
    elif label_type == 'multitask':
        transcript_char = read_text('./sample/LDC93S1.txt')
        transcript_phone = read_phone('./sample/LDC93S1.phn')
        if model == 'ctc':
            transcript_char = ' ' + transcript_char.replace('.', '') + ' '
            labels_char = [alpha2num(transcript_char)] * batch_size
            labels_phone = [phone2num(transcript_phone)] * batch_size
        elif model == 'attention':
            NotImplementedError
        return inputs, labels_char, labels_phone, seq_len

    return inputs, labels, seq_len


def phone2num(transcript):
    """Convert from phone to number.
    Args:
        transcript: sequence of phones (string)
    Returns:
        index_list: list of indices of phone (int)
    """
    phone_list = transcript.split(' ')

    # Read mapping file from phone to number
    phone_dict = {}
    with open('../../experiments/timit/evaluation/mapping_files/ctc/phone2num_61.txt') as f:
        for line in f:
            line = line.strip().split()
            phone_dict[line[0]] = int(line[1])
        phone_dict['<'] = 61
        phone_dict['>'] = 62

    # Convert from phone to the corresponding number
    index_list = []
    for i in range(len(phone_list)):
        if phone_list[i] in phone_dict.keys():
            index_list.append(phone_dict[phone_list[i]])
    return index_list


def num2phone(index_list):
    """Convert from number to phone.
    Args:
        index_list: list of indices of phone (int)
    Returns:
        transcript: sequence of phones (string)
    """
    # Read a phone mapping file
    phone_dict = {}
    with open('../../experiments/timit/evaluation/mapping_files/ctc/phone2num_61.txt') as f:
        for line in f:
            line = line.strip().split()
            phone_dict[int(line[1])] = line[0]
        phone_dict[61] = '<'
        phone_dict[62] = '>'

    # Convert from num to the corresponding phone
    phone_list = []
    for i in range(len(index_list)):
        phone_list.append(phone_dict[index_list[i]])
    transcript = ' '.join(phone_list)
    return transcript


def alpha2num(transcript):
    """Convert from alphabet to number.
    Args:
        transcript: sequence of characters (string)
    Returns:
        index_list: list of indices of alphabet (int)
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
            index_list.append(first_index + 26)  # 27
        elif char == '>':
            index_list.append(first_index + 26 + 1)  # 28
        else:
            index_list.append(ord(char) - first_index)
    return index_list


def num2alpha(index_list):
    """Convert from number to alphabet.
    Args:
        index_list: list of indices of alphabet (int)
    Returns:
        transcript: sequence of character (string)
    """
    # 0 is reserved to space
    first_index = ord('a') - 1
    char_list = []
    for num in index_list:
        if num == 0:
            char_list.append(' ')
        elif num == 27:
            char_list.append('<')
        elif num == 28:
            char_list.append('>')
        else:
            char_list.append(chr(num + first_index))
    transcript = ''.join(char_list)
    return transcript
