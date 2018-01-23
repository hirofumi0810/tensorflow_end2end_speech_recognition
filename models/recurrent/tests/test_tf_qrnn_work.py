#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import functools
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import unittest

sys.path.append(os.pardir)
from qrnn import QRNN


def measure_time(func):
    @functools.wraps(func)
    def _measure_time(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        elapse = time.time() - start
        print("takes {} seconds.".format(elapse))
    return _measure_time


class TestQRNNWork(unittest.TestCase):

    @measure_time
    def test_qrnn(self):
        print("QRNN Working check")
        with tf.Graph().as_default() as qrnn:
            self.check_by_digits(qrnn, qrnn=5)

    @measure_time
    def test_baseline(self):
        print("Baseline(LSTM) Working check")
        with tf.Graph().as_default() as baseline:
            self.check_by_digits(baseline, baseline=True)

    @measure_time
    def test_random(self):
        print("Random Working check")
        with tf.Graph().as_default() as random:
            self.check_by_digits(random, random=True)

    def check_by_digits(self, graph, qrnn=-1, baseline=False, random=False):
        digits = load_digits()
        horizon, vertical, n_class = (8, 8, 10)  # 8 x 8 image, 0~9 number(=10 class)
        size = 128  # state vector size
        batch_size = 128
        images = digits.images / np.max(digits.images)  # simple normalization
        target = np.array([[1 if t == i else 0 for i in range(n_class)]
                           for t in digits.target])  # to 1 hot vector
        learning_rate = 0.001
        train_iter = 1000
        summary_dir = os.path.join(os.path.dirname(__file__), "./summary")

        with tf.name_scope("placeholder"):
            X = tf.placeholder(tf.float32, [batch_size, vertical, horizon])
            y = tf.placeholder(tf.float32, [batch_size, n_class])

        if qrnn > 0:
            pred = self.qrnn_forward(X, size, n_class, batch_size, conv_size=qrnn)
            summary_dir += "/qrnn"
        elif baseline:
            pred = self.baseline_forward(X, size, n_class)
            summary_dir += "/lstm"
        else:
            pred = self.random_forward(X, size, n_class)
            summary_dir += "/random"

        with tf.name_scope("optimization"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        with tf.name_scope("evaluation"):
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.name_scope("summary"):
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("accuracy", accuracy)
            merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(summary_dir, graph)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(train_iter):
                indices = np.random.randint(len(digits.target) - batch_size, size=batch_size)
                _X = images[indices]
                _y = target[indices]
                sess.run(optimizer, feed_dict={X: _X, y: _y})

                if i % 100 == 0:
                    _loss, _accuracy, _merged = sess.run(
                        [loss, accuracy, merged], feed_dict={X: _X, y: _y})
                    writer.add_summary(_merged, i)
                    print("Iter {}: loss={}, accuracy={}".format(i, _loss, _accuracy))

            with tf.name_scope("test-evaluation"):
                acc = sess.run(accuracy, feed_dict={
                               X: images[-batch_size:], y: target[-batch_size:]})
                print("Testset Accuracy={}".format(acc))

    def baseline_forward(self, X, size, n_class):
        shape = X.get_shape()
        # batch_size x sentence_length x word_length -> batch_size x sentence_length x word_length
        _X = tf.transpose(X, [1, 0, 2])
        _X = tf.reshape(_X, [-1, int(shape[2])])  # (batch_size x sentence_length) x word_length
        seq = tf.split(0, int(shape[1]), _X)  # sentence_length x (batch_size x word_length)

        with tf.name_scope("LSTM"):
            lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=1.0)
            outputs, states = rnn.rnn(lstm_cell, seq, dtype=tf.float32)

        with tf.name_scope("LSTM-Classifier"):
            W = tf.Variable(tf.random_normal([size, n_class]), name="W")
            b = tf.Variable(tf.random_normal([n_class]), name="b")
            output = tf.matmul(outputs[-1], W) + b

        return output

    def random_forward(self, X, size, n_class):
        batch_size = int(X.get_shape()[0])

        with tf.name_scope("Random-Classifier"):
            rand_vector = tf.random_normal([batch_size, size])  # batch_size x size random vector
            W = tf.Variable(tf.random_normal([size, n_class]), name="W")
            b = tf.Variable(tf.random_normal([n_class]), name="b")
            output = tf.matmul(rand_vector, W) + b
        return output

    def qrnn_forward(self, X, size, n_class, batch_size, conv_size):
        in_size = int(X.get_shape()[2])

        qrnn = QRNN(in_size=in_size, size=size, conv_size=conv_size)
        hidden = qrnn.forward(X)

        with tf.name_scope("QRNN-Classifier"):
            W = tf.Variable(tf.random_normal([size, n_class]), name="W")
            b = tf.Variable(tf.random_normal([n_class]), name="b")
            output = tf.add(tf.matmul(hidden, W), b)

        return output


if __name__ == "__main__":
    unittest.main()
