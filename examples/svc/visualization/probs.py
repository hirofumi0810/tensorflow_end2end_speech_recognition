#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot posteriors of CTC outputs (SVC corpus)."""

import os
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
sns.set_style("white")

blue = '#4682B4'
orange = '#D2691E'
green = '#006400'

# candidate = ['S2399', 'S1633', 'S1627', 'S2470', 'S1582', 'S2485', 'S1608', 'S1607']
candidate = ['S2399', 'S2171', 'S2470']


def posterior_test(session, posteriors_op, network, dataset, label_type, rate=1.0):
    """Visualize posteriors.
    Args:
        session: session of training model
        posteriois_op: operation for computing posteriors
        network: network to evaluate
        dataset: Dataset class
        label_type: original or phone1 or phone2 or phone41
        rate: rate of evaluation data to use
    """
    # load ground truth labels
    label_dict, _, _ = label.read(
        label_path='/n/sd8/inaguma/corpus/svc/data/labels.txt')

    batch_size = 1
    num_examples = dataset.data_num * rate
    if batch_size == 1 or batch_size % 2 == 0:
        iteration = int(num_examples / batch_size)
    else:
        iteration = int(num_examples / batch_size) + 1

    for step in range(iteration):
        # create feed dictionary for next mini batch
        inputs, labels, seq_len, input_names = dataset.next_batch(
            batch_size=batch_size)
        indices, values, dense_shape = list2sparsetensor(labels)

        feed_dict = {
            network.inputs_pl: inputs,
            network.label_indices_pl: indices,
            network.label_values_pl: values,
            network.label_shape_pl: dense_shape,
            network.seq_len_pl: seq_len,
            network.keep_prob_input_pl: 1.0,
            network.keep_prob_hidden_pl: 1.0
        }

        # visualize
        batch_size_each = len(labels)
        max_frame_num = inputs.shape[1]
        posteriors = session.run(posteriors_op, feed_dict=feed_dict)
        for i_batch in range(batch_size_each):
            posteriors_index = np.array([i_batch + (batch_size_each * j)
                                         for j in range(max_frame_num)])
            label_each = label_dict[input_names[i_batch]]

            # if input_names[i_batch] in candidate:
            #     # np.save(input_names[i_batch], posteriors[posteriors_index])
            #     np.save('label_' + input_names[i_batch], label_each)

            if label_type == 'phone1':
                probs.plot_probs_ctc_phone1(probs=posteriors[posteriors_index],
                                            save_path=network.model_dir,
                                            input_name=input_names[i_batch],
                                            ground_truth_list=label_each)
            elif label_type == 'phone2':
                probs.plot_probs_ctc_phone2(probs=posteriors[posteriors_index],
                                            save_path=network.model_dir,
                                            input_name=input_names[i_batch],
                                            ground_truth_list=label_each)
            elif label_type == 'phone41':
                probs.plot_probs_ctc_phone41(probs=posteriors[posteriors_index],
                                             save_path=network.model_dir,
                                             input_name=input_names[i_batch],
                                             ground_truth_list=label_each)


def plot_probs_framewise(probs, save_path, input_name):
    """Plot posteriors of frame-wise classifiers.
    Args:
        probs: posteriors of each class
        save_path: path to save graph
    """

    # plot probs
    save_path = os.path.join(save_path, '{0:04d}'.format(input_name) + '.png')
    times = np.arange(len(probs)) * 0.01
    plt.clf()
    # plt.figure(figsize=(10, 10))
    plt.subplot(211)
    # plt.plot(times, probs[:, 0],  label='garbage')
    plt.plot(times, probs[:, 1], label='laughter')
    plt.plot(times, probs[:, 2], label='filler')
    # plt.plot(probs[3][0:10], label='blank')
    plt.title('Probs: ' + save_path)
    plt.ylabel('Probability', fontsize=12)
    plt.xlim([0, times[-1]])
    plt.ylim([0, 1])
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    # plot smoothed probs

    # plot waveform
    wav_path = '/n/sd8/inaguma/dataset/SVC/wav/S' + \
        '{0:04d}'.format(input_name) + '.wav'
    sampling_rate, waveform = scipy.io.wavfile.read(wav_path)
    sampling_interval = 1.0 / sampling_rate
    waveform = waveform / 32768.0  # normalize
    times = np.arange(len(waveform)) * sampling_interval
    plt.subplot(212)
    plt.plot(times, waveform, color='grey')
    plt.xlabel('Time[sec]', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.savefig(save_path)
    # plt.show()


def plot_probs_ctc_phone1(probs, save_path, input_name, ground_truth_list):
    """Plot posteriors of phone1.
    Args:
        probs: posteriors of each class
        save_path: path to save graph
        input_name: input name
        ground_truth_list: list of ground truth labels
    """
    if input_name not in candidate:
        return 0

    ####################
    # waveform
    ####################
    wav_path = '/n/sd8/inaguma/corpus/svc/data/wav/' + input_name + '.wav'
    # wav_path = os.path.join(os.path.abspath('../../../../../'), input_name + '.wav')
    sampling_rate, waveform = scipy.io.wavfile.read(wav_path)
    sampling_interval = 1.0 / sampling_rate
    waveform = waveform / 32768.0
    times_waveform = np.arange(len(waveform)) * sampling_interval
    plt.clf()
    # plt.figure(figsize=(10, 5))
    plt.subplot(311)
    plt.title(input_name + '.wav')
    plt.tick_params(labelleft='off')
    plt.tick_params(labelbottom='off')
    plt.plot(times_waveform, waveform, color='grey')
    # plt.xlabel('Time[sec]', fontsize=12)
    plt.ylabel('Input speech', fontsize=12)
    plt.xlim([0, times_waveform[-1]])

    ####################
    # ground truth
    ####################
    times_probs = np.arange(len(probs)) * 0.01
    laughter = np.array([-1] * len(times_probs))
    filler = np.array([-1] * len(times_probs))
    for i in range(len(ground_truth_list)):
        start_frame = int(ground_truth_list[i][1])
        end_frame = int(ground_truth_list[i][2])
        if ground_truth_list[i][0] == 'laughter':
            laughter[start_frame:end_frame] = 1
        elif ground_truth_list[i][0] == 'filler':
            filler[start_frame:end_frame] = 1
    plt.plot(times_probs, laughter, label='laughter (ground truth)',
             color=orange, linewidth=2)
    plt.plot(times_probs, filler, label='filler (ground truth)',
             color=green, linewidth=2)
    plt.legend(loc="upper right", fontsize=12)

    ####################
    # social signals
    ####################
    plt.subplot(312)
    plt.tick_params(labelbottom='off')
    plt.plot(times_probs, probs[:, 1],
             label='laughter (prediction)', color=orange, linewidth=2)
    plt.plot(times_probs, probs[:, 2],
             label='filler (prediction)', color=green, linewidth=2)
    plt.ylabel('Social signals', fontsize=12)
    plt.xlim([0, times_waveform[-1]])
    plt.ylim([0.05, 1.05])
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    ####################
    # garbage
    ####################
    plt.subplot(313)
    plt.tick_params(labelbottom='off')
    line_garbage, = plt.plot(
        times_probs, probs[:, 0], label='garbage', color='black', linewidth=2)
    line_blank, = plt.plot(
        times_probs, probs[:, 3], ':', label='blank', color='grey')
    plt.ylabel('Other classes', fontsize=12)
    plt.xlim([0, times_waveform[-1]])
    plt.ylim([0.05, 1.05])
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(handles=[line_blank, line_garbage],
               loc="upper right", fontsize=12)

    save_path = os.path.join(save_path, input_name + '.png')
    plt.savefig(save_path, dvi=500)
    # plt.show()


def plot_probs_ctc_phone2(probs, save_path, input_name, ground_truth_list):
    """Plot posteriors of phone2.
    Args:
        probs: posteriors of each class
        save_path: path to save graph
        input_name: input name
        ground_truth_list: list of ground truth labels
    """
    if input_name not in candidate:
        return 0

    ####################
    # waveform
    ####################
    wav_path = '/n/sd8/inaguma/corpus/svc/data/wav/' + input_name + '.wav'
    # wav_path = os.path.join(os.path.abspath('../../../../../'), input_name + '.wav')
    sampling_rate, waveform = scipy.io.wavfile.read(wav_path)
    sampling_interval = 1.0 / sampling_rate
    waveform = waveform / 32768.0
    times_waveform = np.arange(len(waveform)) * sampling_interval
    plt.clf()
    # plt.figure(figsize=(10, 5))
    plt.subplot(311)
    plt.title(input_name + '.wav')
    plt.tick_params(labelleft='off')
    plt.tick_params(labelbottom='off')
    plt.plot(times_waveform, waveform, color='grey')
    # plt.xlabel('Time[sec]', fontsize=12)
    plt.ylabel('Input speech', fontsize=12)
    plt.xlim([0, times_waveform[-1]])

    ####################
    # ground truth
    ####################
    times_probs = np.arange(len(probs)) * 0.01
    laughter = np.array([-1] * len(times_probs))
    filler = np.array([-1] * len(times_probs))
    for i in range(len(ground_truth_list)):
        start_frame = int(ground_truth_list[i][1])
        end_frame = int(ground_truth_list[i][2])
        if ground_truth_list[i][0] == 'laughter':
            laughter[start_frame:end_frame] = 1
        elif ground_truth_list[i][0] == 'filler':
            filler[start_frame:end_frame] = 1
    plt.plot(times_probs, laughter, label='laughter (ground truth)',
             color=orange, linewidth=2)
    plt.plot(times_probs, filler, label='filler (ground truth)',
             color=green, linewidth=2)
    plt.legend(loc="upper right", fontsize=12)

    ####################
    # social signals
    ####################
    plt.subplot(312)
    plt.tick_params(labelbottom='off')
    plt.plot(times_probs, probs[:, 1],
             label='laughter (prediction)', color=orange, linewidth=2)
    plt.plot(times_probs, probs[:, 2],
             label='filler (prediction)', color=green, linewidth=2)
    plt.ylabel('Social signals', fontsize=12)
    plt.xlim([0, times_waveform[-1]])
    plt.ylim([0.05, 1.05])
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    ####################
    # speech & silence
    ####################
    plt.subplot(313)
    line_sil, = plt.plot(
        times_probs, probs[:, 0], label='silence', color='black', linewidth=2)
    plt.plot(times_probs, probs[:, 3], label='speech', color=blue, linewidth=2)
    line_blank, = plt.plot(
        times_probs, probs[:, 4], ':', label='blank', color='grey')
    plt.ylabel('Other classes', fontsize=12)
    plt.xlim([0, times_waveform[-1]])
    plt.ylim([0.05, 1.05])
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(handles=[line_blank, line_sil], loc="upper right", fontsize=12)

    save_path = os.path.join(save_path, input_name + '.png')
    plt.savefig(save_path, dvi=500)
    # plt.show()


def plot_probs_ctc_phone41(probs, save_path, input_name, ground_truth_list):
    """Plot posteriors of phone41.
    Args:
        probs: posteriors of each class
        save_path: path to save graph
        input_name: input name
        ground_truth_list: list of ground truth labels
    """
    if input_name not in candidate:
        return 0

    ####################
    # waveform
    ####################
    wav_path = '/n/sd8/inaguma/corpus/svc/data/wav/' + input_name + '.wav'
    # wav_path = os.path.join(os.path.abspath('../../../../../'), input_name + '.wav')
    sampling_rate, waveform = scipy.io.wavfile.read(wav_path)
    sampling_interval = 1.0 / sampling_rate
    waveform = waveform / 32768.0
    times_waveform = np.arange(len(waveform)) * sampling_interval
    plt.clf()
    # plt.figure(figsize=(10, 5))
    plt.subplot(311)
    plt.title(input_name + '.wav')
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    plt.plot(times_waveform, waveform, color='grey')
    # plt.xlabel('Time[sec]', fontsize=12)
    plt.ylabel('Input speech', fontsize=12)
    plt.xlim([0, times_waveform[-1]])

    ####################
    # ground truth
    ####################
    times_probs = np.arange(len(probs)) * 0.01
    laughter = np.array([-1] * len(times_probs))
    filler = np.array([-1] * len(times_probs))
    for i in range(len(ground_truth_list)):
        start_frame = int(ground_truth_list[i][1])
        end_frame = int(ground_truth_list[i][2])
        if ground_truth_list[i][0] == 'laughter':
            laughter[start_frame:end_frame] = 1
        elif ground_truth_list[i][0] == 'filler':
            filler[start_frame:end_frame] = 1
    plt.plot(times_probs, laughter, color=orange,
             linewidth=2, label='laughter (ground truth)')
    plt.plot(times_probs, filler, color=green,
             linewidth=2, label='filler (ground truth)')
    plt.legend(loc="upper right", fontsize=12)

    ####################
    # social signals
    ####################
    plt.subplot(312)
    plt.tick_params(labelbottom='off')
    plt.plot(times_probs, probs[:, 1],
             label='laughter (prediction)', color=orange, linewidth=2)
    plt.plot(times_probs, probs[:, 2],
             label='filler (prediction)', color=green, linewidth=2)
    plt.ylabel('Social signals', fontsize=12)
    plt.xlim([0, times_waveform[-1]])
    plt.ylim([0.05, 1.05])
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(loc="upper right", fontsize=12)

    ####################
    # 41 phones
    ####################
    plt.subplot(313)
    line_sil, = plt.plot(
        times_probs, probs[:, 0], label='silence', color='black', linewidth=2)
    for i in range(3, 43, 1):
        plt.plot(times_probs, probs[:, i])
    line_blank, = plt.plot(
        times_probs, probs[:, 43], ':', label='blank', color='grey')
    plt.ylabel('Other classes', fontsize=12)
    plt.xlim([0, times_waveform[-1]])
    plt.ylim([0.05, 1.05])
    plt.yticks(list(range(0, 2, 1)))
    plt.legend(handles=[line_blank, line_sil], loc="upper right", fontsize=12)

    save_path = os.path.join(save_path, input_name + '.png')
    plt.savefig(save_path, dvi=500)
    plt.show()


if __name__ == '__main__':
    for file in os.listdir(os.path.abspath('./class2')):
        posterior = np.load(os.path.abspath('./class2/' + file))
        input_name = os.path.basename(file).split('.')[0]
        ground_truth_list = np.load(os.path.abspath('./label_' + file))
        plot_probs_ctc_phone2(posterior, save_path=os.path.abspath('./'),
                              input_name=input_name, ground_truth_list=ground_truth_list)
