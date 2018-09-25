# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_cnn.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/17 16:28:47 by msukhare          #+#    #+#              #
#    Updated: 2018/09/25 17:21:58 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import struct as st
import sys
from matplotlib import pyplot as plt
import tensorflow as tf

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
    weights = tf.Variable(tf.truncated_normal(filter, stddev=0.03, seed=5),\
            name=name+"W")
    bias = tf.Variable(tf.truncated_normal(bias, stddev=0.03, seed=5),\
            name=name+"B")
    conv_layer = tf.nn.conv2d(input, weights, s, padding=p, data_format="NCHW")
    conv_layer += bias
    conv_layer = tf.nn.relu(conv_layer)
    return (conv_layer)

def get_new_y(y, batch, nb_class):
    ret = np.zeros((nb_class, batch), dtype=int)
    for i in range(int(batch)):
        ret[y[i][0]][i] = 1
    return (ret)

def main():
    if (len(sys.argv) < 4):
        print("need more file")
        sys.exit()
    X_train, Y_train, m, rows, cols = read_file_idx(sys.argv[1], sys.argv[2])
    X_test, Y_test, m, rows, cols = read_file_idx(sys.argv[3], sys.argv[4])
    #plt.imshow(X[45], interpolation='none', cmap='gray')
    #plt.show()
    X_train = X_train / 255
    X_test = X_test / 255
    tmp = X_train[0:1]
    tmp1 = Y_train[0:1]
    tmp1 = get_new_y(tmp1, 1, 10)
    conv_layer1 = create_conv_layer(tmp, [5, 5, 1, 30], "VALID", [1, 1, 1, 1], [30, 1, 1],\
            "conv_layer1")
    conv_layer1 = tf.nn.pool(conv_layer1, [2, 2], pooling_type="MAX",\
            padding="VALID", data_format="NCHW")
    conv_layer2 = create_conv_layer(conv_layer1, [3, 3, 30, 15], "VALID", [1, 1, 1, 1], [15, 1, 1],\
            "conv_layer1")
    conv_layer2 = tf.nn.pool(conv_layer2, [2, 2], pooling_type="MAX", padding="VALID",\
            data_format="NCHW")
    flatten_l = tf.reshape(conv_layer2, [6000, 1])
    hidden1_w = tf.Variable(tf.truncated_normal([1000, 6000], stddev=0.03))
    hidden1_bias = tf.Variable(tf.truncated_normal([1000, 1], stddev=0.01))
    dense_layer1 = tf.matmul(hidden1_w, flatten_l) + hidden1_bias
    dense_layer1 = tf.nn.relu(dense_layer1)
    hidden2_w = tf.Variable(tf.truncated_normal([500, 1000], stddev=0.03))
    hidden2_bias = tf.Variable(tf.truncated_normal([500, 1], stddev=0.01))
    dense_layer2 = tf.matmul(hidden2_w, dense_layer1) + hidden2_bias
    dense_layer2 = tf.nn.relu(dense_layer2)
    hidden3_w = tf.Variable(tf.truncated_normal([10, 500], stddev=0.03))
    hidden3_bias = tf.Variable(tf.truncated_normal([10, 1], stddev=0.01))
    dense_layer3 = tf.matmul(hidden3_w, dense_layer2) + hidden3_bias
    out_put_l = tf.nn.softmax(dense_layer3)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_layer3, labels=tmp1))
    training_op = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)
    model = tf.global_variables_initializer()
    with tf.session as ss:
        ss.run(model)
        tmp = X_train[start:end]
        tmp1 = Y_train[start:end]
        for i in range(start, end):
            ss.run(training_op, feed_dict={x: tmp, y: tmp1})




if __name__ == "__main__":
    main()
