# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_cnn.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/17 16:28:47 by msukhare          #+#    #+#              #
#    Updated: 2018/10/11 17:08:04 by msukhare         ###   ########.fr        #
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
from cnn_architecture import init_a_random_cnn

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
            dtype=np.float32).reshape((nb_image, rows, cols, 1))
    return (X, Y, nb_image, rows, cols)

def get_new_y(y, batch, nb_class):
    ret = np.zeros((batch, nb_class), dtype=int)
    for i in range(int(batch)):
        ret[i][y[i]] = 1
    return (ret)

def show_graph(nb_epoch, train, cost):
    #linear spline, maybe doesn't adapted to probleme
    #x_smooth = np.linspace(nb_epoch.min(), nb_epoch.max(), 4)
    #y_smooth = sc.interpolate.spline(nb_epoch, train, x_smooth)
    #y_smooth2 = sc.interpolate.spline(nb_epoch, cost, x_smooth)
    #plt.plot(x_smooth, y_smooth)
    #plt.plot(x_smooth, y_smooth2)

    #basic plot without smoothing
   # plt.plot(nb_epoch, train)
   # plt.plot(nb_epoch, cost)
    #polynome du 3eme degre
#    poly = np.polyfit(nb_epoch, train, 3)
#    poly_y = np.poly1d(poly)(nb_epoch)
#    poly1 = np.polyfit(nb_epoch, cost, 3)
#    poly_y1 = np.poly1d(poly1)(nb_epoch)
#    plt.plot(nb_epoch, poly_y)
#    plt.plot(nb_epoch, poly_y1)
    plt.plot(nb_epoch, train)
    plt.plot(nb_epoch, cost)
    plt.show()

def get_class(pred):
    maxi = pred[0][0]
    print(pred)
    index = 0
    for i in range(10):
        if (pred[0][i] > maxi):
            index = i
            maxi = pred[0][i]
    return (index)

def train_modele2(cross_entropy, out_put, X_train, Y_train, x, y, X_cost, Y_cost, X_test, Y_test, m):
    init = tf.global_variables_initializer()
    learning_rate = tf.placeholder(tf.float32, shape=[])
    training = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    #training = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    epoch = 20
    batch = 128
    alpha = 0.1
    nb_epoch = np.zeros((epoch), dtype=float)
    cost = np.zeros((epoch), dtype=float)
    train = np.zeros((epoch), dtype=float)
    #init = tf.initialize_all_variables()
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(init)
        seed(465)
        for i in range(epoch):
            avg = 0
            avg_cost = 0
            for j in range(0, floor(m * 0.8), batch):
                X = X_train[j: (j + batch)]
                Y = get_new_y(Y_train[j: (j + batch)], batch, 10)
                sess.run(training, feed_dict={learning_rate: alpha, x: X, y: Y})
                if (j < (floor(m * 0.2) - batch)):
                    X_c = X_cost[j: (j + batch)]
                    Y_c = get_new_y(Y_cost[j: (j + batch)], batch, 10)
                    c = sess.run(cross_entropy, feed_dict={x: X_c, y: Y_c})
                    c_t = sess.run(cross_entropy, feed_dict={x: X, y: Y})
                    avg_cost += c
                    avg += c_t
            cost[i] = avg_cost / floor(m * 0.2)
            nb_epoch[i] = i
            train[i] = avg / floor(m * 0.2)
            print("epoch= ", i, "cost_train= ", train[i], "cost_test= ", cost[i])
        good_pred = 0
        for i in range(10000):
            X = np.reshape(X_test[i], (1, 28, 28, 1))
            Y_t = get_new_y(Y_test[i], 1, 10)
            classe = get_class(sess.run(out_put, feed_dict={x: X, y: Y_t}))
            #if ((i % 500) == 0):
                #plt.imshow(np.reshape(X_test[i], (28,28)), interpolation='none', cmap='gray')
                #plt.title("predicted class {0}".format(classe))
                #plt.show()
            if (classe == Y_test[i]):
                good_pred += 1
        print("accuracy: ", good_pred / 10000)
        print("modele has been save in ", saver.save(sess, './tmp/my_model.ckpt'))
        show_graph(nb_epoch, train, cost)

def main():
    if (len(sys.argv) < 4):
        print("need more file")
        sys.exit()
    X_train, Y_train, m, rows, cols = read_file_idx(sys.argv[1], sys.argv[2])
    X_test, Y_test, m_test, rows, cols = read_file_idx(sys.argv[3], sys.argv[4])
    X_train = X_train / 255
    X_test = X_test / 255
    X_cost = X_train[floor(m * 0.8):]
    Y_cost = Y_train[floor(m * 0.8):]
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
   # cross_entropy, out_put = init_a_random_cnn(x, y)
    cross_entropy, out_put = init_net5(x, y)
    train_modele2(cross_entropy, out_put, X_train, Y_train, x, y, X_cost, Y_cost, X_test, Y_test, m)

if __name__ == "__main__":
    main()
