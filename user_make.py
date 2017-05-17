# -*- coding: utf-8 -*-
import sys
from collections import OrderedDict
import numpy as np
import tensorflow as tf

from BayesianOpt import BayesianOpt

# data mnist
from mnist_data.mnist import download_mnist, load_mnist, key_file
download_mnist()
X_train = load_mnist(key_file["train_img"])[8:, :]
X_test = load_mnist(key_file["test_img"], )[8:,:]
y_train = load_mnist(key_file["train_label"], 1)
y_test = load_mnist(key_file["test_label"], 1)

# 目的関数（ハイパーパラメータを引数にする関数）: userに書いてもらう
# (引数: ハイパーパラメータ、戻り値: placeholder, train_step, accuracy)
def MLP(alpha, lr, layer1, layer2, layer3):
    X = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.int32, [None, ])
    y_ = tf.one_hot(label, depth=10, dtype=tf.float32)

    w_0 = tf.Variable(tf.random_normal([784, int(layer1)], mean=0.0, stddev=0.05))
    b_0 = tf.Variable(tf.zeros([int(layer1)]))
    h_0 = tf.sigmoid(tf.matmul(X, w_0) + b_0)

    w_1 = tf.Variable(tf.random_normal([int(layer1), int(layer2)], mean=0.0, stddev=0.05))
    b_1 = tf.Variable(tf.zeros([int(layer2)]))
    h_1 = tf.sigmoid(tf.matmul(h_0, w_1) + b_1)

    w_2 = tf.Variable(tf.random_normal([int(layer2), int(layer3)], mean=0.0, stddev=0.05))
    b_2 = tf.Variable(tf.zeros([int(layer3)]))
    h_2 = tf.sigmoid(tf.matmul(h_1, w_2) + b_2)

    w_o = tf.Variable(tf.random_normal([int(layer3), 10], mean=0.0, stddev=0.05))
    b_o = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(h_2, w_o) + b_o)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    L2_sqr = tf.nn.l2_loss(w_0) + tf.nn.l2_loss(w_1) + tf.nn.l2_loss(w_2)

    loss = cross_entropy + alpha * L2_sqr
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return X, label, train_step, accuracy

if __name__ == "__main__":
    hyper_parameter_range = {"alpha": (1e-8, 1e-4), "lr": (1e-6, 1e-2),
                             "layer1": (10, 100),"layer2": (10, 100),"layer3": (10, 100)}
    hyper_parameter_explore = {"alpha": [1e-8, 1e-8, 1e-4, 1e-4],"lr": [1e-6, 1e-2, 1e-6, 1e-2],
                               "layer1": [10, 50, 100, 50], "layer2": [10, 50, 100, 50],"layer3": [10, 50, 100, 50]}
    BO = BayesianOpt(MLP, hyper_parameter_range, hyper_parameter_explore,
                     X_train, X_test, y_train, y_test)
    BO.main()
