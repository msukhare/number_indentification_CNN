# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_cnn.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/17 16:28:47 by msukhare          #+#    #+#              #
#    Updated: 2018/11/11 16:50:48 by kemar            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import struct as st
from math import floor
import sys
from matplotlib import pyplot as plt
import tensorflow as tf
from random import *
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

def show_graph(nb_epoch, train, test):
    #linear spline, maybe doesn't adapted to probleme
    #x_smooth = np.linspace(nb_epoch.min(), nb_epoch.max(), 4)
    #y_smooth = sc.interpolate.spline(nb_epoch, train, x_smooth)
    #y_smooth2 = sc.interpolate.spline(nb_epoch, cost, x_smooth)
    #plt.plot(x_smooth, y_smooth)
    #plt.plot(x_smooth, y_smooth2)

    #polynome du 3eme degre
    #poly = np.polyfit(nb_epoch, train, 3)
    #poly_y = np.poly1d(poly)(nb_epoch)
    #poly1 = np.polyfit(nb_epoch, cost, 3)
    #poly_y1 = np.poly1d(poly1)(nb_epoch)
    #plt.plot(nb_epoch, poly_y)
    #plt.plot(nb_epoch, poly_y1)

    #basic plot without smoothing
    plt.plot(nb_epoch, train, color='red')
    plt.plot(nb_epoch, test, color='green')
    plt.show()

def get_class(pred):
    maxi = pred[0][0]
    index = 0
    for i in range(10):
        if (pred[0][i] > maxi):
            index = i
            maxi = pred[0][i]
    return (index)

def init_para_tensorflow(cross_entropy):
    learning_rate = tf.placeholder(tf.float32, shape=[])
    training = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    #training = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)
    return (learning_rate, init, training, saver)

def get_accuracy(sess, out_put, X_val, Y_val, x, y):
    good_pred = 0
    for i in range(10000):
        X = np.reshape(X_val[i], (1, 28, 28, 1))
        Y = get_new_y(Y_val[i], 1, 10)
        classe = get_class(sess.run(out_put, feed_dict={x: X, y: Y}))
        #if ((i % 500) == 0):
            #plt.imshow(np.reshape(X_test[i], (28,28)), interpolation='none', cmap='gray')
            #plt.title("predicted class {0}".format(classe))
            #plt.show()
        if (classe == Y_val[i]):
            good_pred += 1
    print("accuracy: ", good_pred / 10000, "good_pred=", good_pred)

def perform_one_epoch(sess, cross_entropy, training, X_train, Y_train, x, y, learning_rate, alpha, m,\
        batch):
    avg = 0
    for i in range(0, m, batch):
        X = X_train[i: (i + batch)]
        Y = get_new_y(Y_train[i: (i + batch)], batch, 10)
        _, tmp = sess.run([training, cross_entropy], feed_dict={learning_rate: alpha, x: X, y: Y})
        avg += tmp
    return (avg / (m / batch))

def perform_one_epoch_rand(sess, cross_entropy, training, X_train, Y_train, x, y, learning_rate,\
        alpha, m, batch):
    avg = 0
    for i in range((m / batch)):
        ind = np.random.randint(0, (m - batch))
        X = X[ind: (ind + batch)]
        Y = get_new_y(Y_train[ind: (ind + batch)], batch, 10)
        _, tmp = sess.run([training, cross_entropy], feed_dict={learning_rate: alpha, x: X, y: Y})
        avg += tmp
    return (avg / (m / batch))

def perform_cost_fun(cross_entropy, sess, x, y, X_test, Y_test, m, batch):
    avg = 0
    j = 0
    for i in range(0, (floor(m * 0.20) - batch), batch):
        X = X_test[i: (i + batch)]
        Y = get_new_y(Y_test[i: (i + batch)], batch, 10)
        avg += sess.run(cross_entropy, feed_dict={x: X, y: Y})
        j += 1
    return (avg / j)

def train_modele(cross_entropy, out_put, X_train, Y_train, x, y, X_test, Y_test, X_val, Y_val, m):
    learning_rate, init, training, saver = init_para_tensorflow(cross_entropy)
    #12 epoch, 128 batch 48k / 128 iter 0.1 alpha Gradient_descent Lenet5 ===> 98.31% accuracy
    #33 epoch, 128 batch 48k / 128 iter 0.1 alpha Gradient_descent Lenet5 ===> 98.51% accuracy
    #100 epoch, 128 batch 48k / 128 iter 0.1 alpha Gradient_descent Lenet5 ===> 98.77% accuracy
    #300 epoch, 128 batch 48k / 128 iter 0.1 alpha Gradient_descent Lenet5 ===> 98.78% accuracy
    #12 epoch, 128 batch 48k / 128 iter 0.01 alpha Gradient_descent Lenet5 ===> 55.79% accuracy
    #100 epoch, 128 batch 48k / 128 iter 0.01 alpha Gradient_descent Lenet5 ===> 88.43% accuracy
    #300 epoch, 128 batch 48k / 128 iter 0.01 alpha Gradient_descent Lenet5 ===> 88% accuracy
    #300 epoch, 128 batch 48k / 128 iter 0.001 alpha Gradient_descent Lenet5 ===> 58.73% accuracy
    #300 epoch, 128 batch 48k / 128 iter 0.1 alpha and decrease of 0.03 after each 12 epoch Gradient_descent Lenet5 ===> 98,67% accuracy
    #12 epoch, 128 batch 48k / 128 iter 0.1 alpha Gradient_descent random ===> 94.16% accuracy
    #100 epoch, 128 batch 48k / 128 iter 0.1 alpha Gradient_descent random ===> 96.7% accuracy
    #300 epoch, 128 batch 48k / 128 iter 0.1 alpha Gradient_descent random ===> 97.02% accuracy
    #12 epoch, 128 batch 48k / 128 iter 0.01 alpha Gradient_descent random ===> 36.13% accuracy
    #100 epoch, 128 batch 48k / 128 iter 0.01 alpha Gradient_descent random ===> 94.21% accuracy
    #300 epoch, 128 batch 48k / 128 iter 0.01 alpha Gradient_descent random ===> 96.49% accuracy
    #300 epoch, 128 batch 48k / 128 iter 0.001 alpha Gradient_descent random ===> 80.80% accuracy
   #300 epoch, 128 batch 48k / 128 iter 0.1 alpha and decrease of 0.03 after each 12 epoch Gradient_descent Lenet5 ===> 95,15% accuracy
    #with adamoptimizer
    #100 epoch, 128 batch 48k / 128 iter 0.1 alpha ADAM random ===> 9.8% accuracy
    #100 epoch, 128 batch 48k / 128 iter 0.01 alpha ADAM random ===> 76.11% accuracy
    #300 epoch, 128 batch 48k / 128 iter 0.001 alpha ADAM random ===> 97.52% accuracy
    epoch = 200
    batch  = 32
    alpha = 0.1
    nb_epoch = np.zeros((epoch), dtype=float)
    test = np.zeros((epoch), dtype=float)
    train = np.zeros((epoch), dtype=float)
    with tf.Session() as sess:
        sess.run(init)
        seed(465)
        for i in range(epoch):
            train[i] = perform_one_epoch(sess, cross_entropy, training, X_train, Y_train, x, y, \
                    learning_rate, alpha, floor(m * 0.80), batch)
           # train[i] = perform_one_epoch_rand(sess, cross_entropy, training, X_train, Y_train, x,\
            #        y, learning_rate, alpha, floor(m * 0.80), batch)
            test[i] = perform_cost_fun(cross_entropy, sess, x, y, X_test, Y_test, m, batch)
            nb_epoch[i] = i
            #if (((i + 1) % 12) == 0):
                #alpha *= 0.03
            print("epoch= ", i, "cost_train= ", train[i], "cost_test= ", test[i])
        get_accuracy(sess, out_put, X_test, Y_test, x, y)
        print("model has been save in ", saver.save(sess, './tmp/my_model.ckpt'))
        show_graph(nb_epoch, train, test)

def main():
    if (len(sys.argv) < 4):
        print("need more file")
        sys.exit()
    X_train, Y_train, m, rows, cols = read_file_idx(sys.argv[1], sys.argv[2])
    X_val, Y_val, m_test, rows, cols = read_file_idx(sys.argv[3], sys.argv[4])
    X_train = X_train / 255
    X_val = X_val / 255
    X_test = X_train[floor(m * 0.8):]
    Y_test = Y_train[floor(m * 0.8):]
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="y")
    #cross_entropy, out_put = init_a_random_cnn(x, y)
    cross_entropy, out_put = init_net5(x, y)
    train_modele(cross_entropy, out_put, X_train, Y_train, x, y, X_test, Y_test, X_val, Y_val, m)

if __name__ == "__main__":
    main()
