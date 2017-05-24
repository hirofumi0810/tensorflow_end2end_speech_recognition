#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.io.wavfile
from python_speech_features import mfcc, fbank, logfbank, hz2mel


def read_wav(feature_type='logmelfbank'):
    """
    Args:
        feature: logmelfbank or mfcc
    Returns:
        inputs:
        seq_len:
    """
    # load wav file
    wav_path = './sample/LDC93S1.wav'
    fs, audio = scipy.io.wavfile.read(wav_path)

    # convert to mfcc features
    if feature_type == 'mfcc':
        features = mfcc(audio, samplerate=fs)  # (291, 13)

    # convert to log-mel fbank & log energy
    elif feature_type == 'logmelfbank':
        fbank_features, energy = fbank(audio, nfilt=40)
        logfbank = np.log(fbank_features)
        logenergy = np.log(energy)
        logmelfbank = hz2mel(logfbank)
        features = np.c_[logmelfbank, logenergy]  # (291, 41)

    delta1 = delta(features, N=2)
    delta2 = delta(delta1, N=2)
    inputs = np.c_[features, delta1, delta2]

    # transform to 3D array
    inputs = np.asarray(inputs[np.newaxis, :])  # (1, 291, 39) or (1, 291, 123)
    seq_len = [inputs.shape[1]]  # [291]

    # normalization
    inputs = (inputs - np.mean(inputs)) / np.std(inputs)

    # save as npy file
    np.save('./sample/inputs.npy', inputs)
    np.save('./sample/seq_len.npy', seq_len)
    return inputs, seq_len


def delta(feat, N):
    """Compute delta features from a feature vector sequence.
    Args:
        feat: A numpy array of size (NUMFRAMES by number of features) containing features.
            Each row holds 1 feature vector.
        N: For each frame, calculate delta features based on preceding and following N frames.
    Rreturns:
        A numpy array of size (NUMFRAMES by number of features) containing delta features.
            Each row holds 1 delta feature vector.
    """
    NUMFRAMES = len(feat)
    feat = np.concatenate(([feat[0] for i in range(N)], feat, [feat[-1] for i in range(N)]))
    denom = sum([2 * i * i for i in range(1, N + 1)])
    dfeat = []
    for j in range(NUMFRAMES):
        dfeat.append(np.sum([n * feat[N + j + n] for n in range(-1 * N, N + 1)], axis=0) / denom)
    return dfeat


def read_text():
    """
    Returns:
        labels: list of alphabet index
    """
    # read target labels
    text_path = './sample/LDC93S1.txt'
    with open(text_path, 'r') as f:
        line = f.readlines()[-1]
        text = ' ' + ' '.join(line.strip().lower().split(' ')[2:]).replace('.', '') + ' '

    # convert form alphabet to number
    char_list = list(text)
    index_list = alpha2num(char_list)
    labels = [index_list]

    # save as npy file
    np.save('./sample/labels.npy', labels)
    return labels


def alpha2num(char_list):
    """Convert from alphabet to number.
    Args:
        char_list: list of character (string)
    Returns:
        num_list: list of alphabet index (int)
    """
    # 0 is reserved for space
    space_index = 0
    first_index = ord('a') - 1
    num_list = [space_index if char == ' ' else ord(char) - first_index for char in char_list]
    return num_list


def num2alpha(index_list):
    """Convert from number to alphabet.
    Args:
        index_list: list of alphabet index (int)
    Returns:
        char_list: list of character (string)
    """
    # 0 is reserved to space
    first_index = ord('a') - 1
    char_list = [' ' if num == 0 else chr(num + first_index) for num in index_list]
    return char_list


def read_phone(label_type):
    if os.path.isfile('./sample/labels_.' + label_type + 'npy'):
        labels = np.load('./sample/labels_.' + label_type + 'npy')
    else:
        # read target labels
        phone_path = './sample/LDC93S1.phn'
        phone_list = []
        with open(phone_path, 'r') as f:
            for line in f:
                line = line.strip().split(' ')
                phone_list.append(line[-1])

        # convert form phone to number
        index_list = phone2num(phone_list, phone_num=int(label_type[5:7]))
        labels = [index_list]

        # save as npy file
        np.save('./sample/labels_' + label_type + '.npy', labels)
    return labels


def phone2num(phone_list, phone_num):
    """Convert from phone to number.
    Args:
        phone_list: list of phones (string)
        phone_num: 39 or 48 or 61
    Returns:
        phone_list: list of phone indices (int)
    """
    map_dict = {}
    if os.path.isfile(os.path.join(os.path.abspath('../../../../prepare/timit/'),
                                   'phone2num_' + str(phone_num) + '.txt')):
        # read mapping file (from phone to num)
        with open(os.path.join(os.path.abspath('../../../../prepare/timit/'),
                               'phone2num_' + str(phone_num) + '.txt')) as f:
            for line in f:
                line = line.strip().split()
                map_dict[line[0]] = int(line[1])

    # convert from phone to number
    for i in range(len(phone_list)):
        if phone_list[i] in map_dict.keys():
            phone_list[i] = map_dict[phone_list[i]]
    return phone_list


def num2phone(num_list, phone_num):
    """Convert from number to phone.
    Args:
        num_list: list of phone indices (int)
        phone_num: 39 or 48 or 61
    Returns:
        phone_list: list of phones (string)
    """
    # read a phone mapping file
    phone_dict = {}
    if phone_num == 39:
        mapping_file_name = 'phone2num_39.txt'
    elif phone_num == 48:
        mapping_file_name = 'phone2num_48.txt'
    elif phone_num == 61:
        mapping_file_name = 'phone2num_61.txt'
    with open(os.path.join(os.path.abspath('../../../../prepare/timit/'), mapping_file_name)) as f:
        for line in f:
            line = line.strip().split()
            phone_dict[int(line[1])] = line[0]

    # convert from num to the corresponding phones
    phone_list = []
    for i in range(len(num_list)):
        phone_list.append(phone_dict[num_list[i]])

    return phone_list


def list2sparsetensor(labels):
    """Convert labels from list to sparse tensor.
    Args:
        labels: list of labels
    Returns:
        labels_st: sparse tensor of labels, list of indices, values, dense_shape
    """
    indices, values = [], []
    for i_utt, each_label in enumerate(labels):
        for i_l, l in enumerate(each_label):
            indices.append([i_utt, i_l])
            values.append(l)
    dense_shape = [len(labels), np.asarray(indices).max(0)[1] + 1]
    labels_st = [np.array(indices), np.array(values), np.array(dense_shape)]
    return labels_st


def sparsetensor2list(labels_st, batch_size):
    """Convert labels from sparse tensor to list.
    Args:
        labels_st: sparse tensor of labels
        batch_size: batch size
    Returns:
        labels: list of labels
    """
    indices = labels_st.indices
    values = labels_st.values

    labels = []
    batch_boundary = np.where(indices[:, 1] == 0)[0]

    # print(batch_boundary)
    # if len(batch_boundary) != batch_size:
    #     batch_boundary = np.array(batch_boundary.tolist() + [max(batch_boundary) + 1])
    # print(indices)

    for i in range(batch_size - 1):
        label_each_wav = values[batch_boundary[i]:batch_boundary[i + 1]]
        labels.append(label_each_wav.tolist())
    labels.append(values[batch_boundary[-1]:].tolist())
    return labels


def generate_batch(label_type):

    if not os.path.isfile('./inputs.npy') or not os.path.isfile('./seq_len.npy'):
        inputs, seq_len = read_wav(feature_type='logmelfbank')
        # inputs, seq_len = read_wav(feature_type='mfcc')
    else:
        inputs = np.load('./sample/inputs.npy')
        seq_len = np.load('./sample/seq_len.npy')

    if label_type == 'character':
        if os.path.isfile('./sample/labels_character.npy'):
            labels = np.load('./sample/labels_character.npy')
        else:
            labels = read_text()

    elif label_type[:5] == 'phone':
        labels = read_phone(label_type)

    return inputs, labels, seq_len


if __name__ == '__main__':
    generate_batch(label_type='phone')
    generate_batch(label_type='character')
