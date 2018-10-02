# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_cnn.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/17 16:28:47 by msukhare          #+#    #+#              #
#    Updated: 2018/10/02 16:40:10 by kemar            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import struct as st
from math import floor
import sys
from matplotlib import pyplot as plt
import tensorflow as tf
from random import *
from numpy.linalg import inv
import scipy as sc
def read_file_idx(images, labels):
    try:
        image_file = open(images, "rb")
    except:
        print(images, "doesn't exist")
        sys.exit()
    try:
        label_file = open(labels, "rb")
    except:
        print(labels, "doesn't exist")
        sys.exit
    magic_image, nb_image, rows, cols = st.unpack('>IIII',image_file.read(16))
    magic_label, nb_labels = st.unpack('>II', label_file.read(8))
    Y = np.asarray(st.unpack('>'+'B'*nb_labels, label_file.read(nb_labels))).reshape(nb_labels, 1)
    byte_read = (rows * cols * nb_image * 1)
    X = np.asarray(st.unpack('>'+'B'*byte_read, image_file.read(byte_read)),\
            dtype=np.float32).reshape((nb_image, 1, rows, cols))
    return (X, Y, nb_image, rows, cols)

def create_conv_layer(input, filter, p, s, bias, name):
    weights = tf.Variable(tf.truncated_normal(filter, stddev=0.05, seed=5),\
            name=name+"W")
    bias = tf.Variable(tf.truncated_normal(bias, stddev=0.05, seed=5),\
            name=name+"B")
    conv_layer = tf.nn.conv2d(input, weights, s, padding=p)
    conv_layer += bias
    #conv_layer = tf.nn.relu(conv_layer)
    return (conv_layer)

def define_dense_layers(in_put, nb_before, nb_neurones):
    weights = tf.Variable(tf.truncated_normal([nb_before, nb_neurones],\
            stddev=0.05, seed=5), name="weights1")
    bias = tf.Variable(tf.truncated_normal([1, nb_neurones], stddev=0.05, seed=5),\
            name="bias1")
    return (tf.nn.relu(tf.matmul(in_put, weights) + bias))

def get_new_y(y, batch, nb_class):
    ret = np.zeros((batch, nb_class), dtype=int)
    for i in range(int(batch)):
        ret[i][y[i]] = 1
    return (ret)

def define_conv_op(x, y):
    layer1 = create_conv_layer(x, [5, 5, 1, 6], "VALID", [1, 1, 1, 1],\
            [1, 1, 6], "first_conv_layer")
    layer1 = tf.nn.pool(layer1, [2, 2], pooling_type="MAX",\
            padding="VALID", strides=[2, 2])
    layer2 = create_conv_layer(layer1, [5, 5, 6, 16], "VALID", [1, 1, 1, 1],\
            [1, 1, 16], "sec_conv_layer")
    layer2 = tf.nn.pool(layer2, [2, 2], pooling_type="MAX",\
            padding="VALID", strides=[2, 2])
    return (layer2)

def init_net5(x, y):
    out_conv = define_conv_op(x, y)
    flatten_layer = tf.reshape(out_conv, [-1, (out_conv.shape[1] *\
            out_conv.shape[2] * out_conv.shape[3])])
    out_dense = define_dense_layers(flatten_layer, 256, 120)
    out_dense = define_dense_layers(out_dense, 120, 84)
    out_dense = define_dense_layers(out_dense, 84, 10)
    return (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(\
            logits=out_dense, labels=y)))

def spline(X, Y):
    h = np.zeros(((len(X) - 1), 1), dtype=float)
    for i in range((len(X) - 1)):
        h[i][0] = X[i + 1] - X[i]
    f = np.zeros((len(X), 1), dtype=float)
    for i in range(1, (len(X) - 1)):
        tmp = (Y[i + 1] - Y[i]) / h[i][0]
        tmp1 = (Y[i] - Y[i - 1]) / h[i - 1]
        f[i][0] = tmp - tmp1
    r = np.zeros((len(X), len(X)), dtype=float)
    r[0][0] = 1
    r[(len(X) - 1)][(len(X) - 1)] = 1
    for i in range(1, (len(X) - 1)):
        r[i][i] = (h[i - 1][0] + h[i][0]) / 3
    for i in range(1, (len(X) - 1)):
        r[i][i + 1] = h[i][0] / 6
        r[i][i - 1] = h[i][0] / 6
    m = inv(r).dot(f)
    c = np.zeros(((len(X) - 1), 1), dtype=float)
    cprime = np.zeros(((len(X) - 1), 1), dtype=float)
    for i in range((len(X) - 1)):
        c[i][0] = (Y[i + 1] - Y[i]) / h[i][0] - h[i] / 6 * (m[i + 1][0] - m[i][0])
        cprime[i][0] = Y[i] - m[i][0] * ((h[i][0]**2) / 6)
    print(h, "\n\n", f, "\n\n", r, "\n\n", m, "\n\n", c, "\n\n", cprime)

def train_modele(cross_entropy, X_train, Y_train, x, y, X_cost, Y_cost, m):
    init = tf.global_variables_initializer()
    #learning_rate = tf.Variable(0.005, tf.float32)
    training = tf.train.GradientDescentOptimizer(0.0005).minimize(cross_entropy)
    epoch = 5000
    nb_epoch = np.zeros((epoch), dtype=float)
    cost = np.zeros((epoch), dtype=float)
    train = np.zeros((epoch), dtype=float)
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epoch):
            start = randint(0, (floor(m * 0.8) - 32))
            start2 = randint(0, (floor(m * 0.2) - 32))
            avg = 0
            avg_cost = 0
            for j in range(32):
                X = np.reshape(X_train[(start + j)], (1, 28, 28, 1))
                X_c = np.reshape(X_cost[(start2 + j)], (1, 28, 28, 1))
                Y = get_new_y(Y_train[(start + j)], 1, 10)
                Y_c = get_new_y(Y_cost[(start2 + j)], 1, 10)
                _, c = sess.run([training, cross_entropy],\
                        feed_dict={x: X, y: Y})
                c  = sess.run(cross_entropy, feed_dict={x: X_c, y: Y_c})
                avg_cost += c / 32
                avg += c / 32
            cost[i] = avg_cost
            nb_epoch[i] = i
            train[i] = avg
            print("epoch = ", i, "cost = ", avg)
        sess.close()
    x_smooth = np.linspace(nb_epoch.min(), nb_epoch.max(), (epoch / 2))
    y_smooth = sc.interpolate.spline(nb_epoch, train, x_smooth)
    plt.plot(x_smooth, y_smooth)
    plt.show()

def main():
    if (len(sys.argv) < 4):
        print("need more file")
        sys.exit()
   # spline([0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 0.4794, 0.8415, 0.9975, 0.9093])
   # sys.exit()
    X_train, Y_train, m, rows, cols = read_file_idx(sys.argv[1], sys.argv[2])
    X_test, Y_test, m_test, rows, cols = read_file_idx(sys.argv[3], sys.argv[4])
    #plt.imshow(X[45], interpolation='none', cmap='gray')
    #plt.show()
    X_train = X_train / 255
    X_test = X_test / 255
    X_cost = X_train[floor(m * 0.8):]
    Y_cost = Y_train[floor(m * 0.8):]
    x = tf.placeholder(tf.float32, shape=[1, 28, 28, 1])
    y = tf.placeholder(tf.float32, shape=[1, 10])
    cross_entropy = init_net5(x, y)
    #graph = tf.get_default_graph()
    #for op in graph.get_operations():
     #   print(op.name)
    train_modele(cross_entropy, X_train, Y_train, x, y, X_cost, Y_cost, m)

if __name__ == "__main__":
    main()
