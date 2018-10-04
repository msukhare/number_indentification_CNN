# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_cnn.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/17 16:28:47 by msukhare          #+#    #+#              #
#    Updated: 2018/10/04 17:11:29 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import struct as st
from math import floor
import sys
from matplotlib import pyplot as plt
import tensorflow as tf
from random import *
import scipy as sc
from cnn_architecture import init_net5

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
            dtype=np.float32).reshape((nb_image, 1, rows, cols, 1))
    return (X, Y, nb_image, rows, cols)

def get_new_y(y, batch, nb_class):
    ret = np.zeros((batch, nb_class), dtype=int)
    for i in range(int(batch)):
        ret[i][y[i]] = 1
    return (ret)

def show_graph(nb_epoch, train, cost):
    x_smooth = np.linspace(nb_epoch.min(), nb_epoch.max(), 300)
    y_smooth = sc.interpolate.spline(nb_epoch, train, x_smooth)
    y_smooth2 = sc.interpolate.spline(nb_epoch, cost, x_smooth)
    plt.plot(x_smooth, y_smooth)
    plt.plot(x_smooth, y_smooth2)
    print("all ok")
    plt.show()

def get_class(pred):
    maxi = pred[0][0]
    index = 0
    for i in range(10):
        if (pred[0][i] > maxi):
            index = i
            maxi = pred[0][i]
    return (index)

def train_modele(cross_entropy, out_put, X_train, Y_train, x, y, X_cost, Y_cost, X_test, Y_test, m):
    init = tf.global_variables_initializer()
    #learning_rate = tf.Variable(0.005, tf.float32)
    training = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    epoch = 300
    batch = 64
    nb_epoch = np.zeros((epoch), dtype=float)
    cost = np.zeros((epoch), dtype=float)
    train = np.zeros((epoch), dtype=float)
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epoch):
            start = randint(0, (floor(m * 0.8) - batch))
            start2 = randint(0, (floor(m * 0.2) - batch))
            avg = 0
            avg_cost = 0
            accu = 0
            for j in range(batch):
                Y = get_new_y(Y_train[(start + j)], 1, 10)
                Y_c = get_new_y(Y_cost[(start2 + j)], 1, 10)
                _, c = sess.run([training, cross_entropy],\
                        feed_dict={x: X_train[(start + j)], y: Y})
                avg += c
                c  = sess.run(cross_entropy, feed_dict={x: X_cost[(start2 + j)], y: Y_c})
                avg_cost += c
            cost[i] = avg_cost / batch
            nb_epoch[i] = i
            train[i] = avg / batch
            print("epoch= ", i, "cost_train= ", train[i], "cost_test= ", cost[i])
            if (train[i] <= 0.09):
                break
        good_pred = 0
        for i in range(10000):
            Y_t = get_new_y(Y_test[i], 1, 10)
            if (get_class(sess.run(out_put, feed_dict={x: X_test[i], y: Y_t})) == Y_test[i]):
                good_pred += 1
        sess.close()
    print("accuracy: ", good_pred / 10000)
    show_graph(nb_epoch, train, cost)

def main():
    if (len(sys.argv) < 4):
        print("need more file")
        sys.exit()
    X_train, Y_train, m, rows, cols = read_file_idx(sys.argv[1], sys.argv[2])
    X_test, Y_test, m_test, rows, cols = read_file_idx(sys.argv[3], sys.argv[4])
    #show handwritten number
    #plt.imshow(X[45], interpolation='none', cmap='gray')
    #plt.show()
    X_train = X_train / 255
    X_test = X_test / 255
    X_cost = X_train[floor(m * 0.8):]
    Y_cost = Y_train[floor(m * 0.8):]
    x = tf.placeholder(tf.float32, shape=[1, 28, 28, 1])
    y = tf.placeholder(tf.float32, shape=[1, 10])
    cross_entropy, out_put = init_net5(x, y)
    #graph = tf.get_default_graph()  put graphe tensorflow
    #for op in graph.get_operations():
     #   print(op.name)
    train_modele(cross_entropy, out_put, X_train, Y_train, x, y, X_cost, Y_cost, X_test, Y_test, m)

if __name__ == "__main__":
    main()
