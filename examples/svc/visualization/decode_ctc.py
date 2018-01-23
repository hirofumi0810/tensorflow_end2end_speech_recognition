def decode_test(session, decode_op, network, dataset, label_type, rate=1.0):
    """Visualize label outputs.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: Dataset class
        label_type: original or phone1 or phone2 or phone41
        rate: rate of evaluation data to use
    """
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
        labels_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = dataset.sparsetensor2list(labels_st, batch_size_each)
        for i_batch in range(batch_size_each):
            print('-----wav: %s-----' % input_names[i_batch])
            print('Pred: ', end="")
            print(np.array(labels_pred[i_batch]))
            print('True: ', end="")
            print(labels[i_batch])
