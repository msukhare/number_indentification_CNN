# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_cnn.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/17 16:28:47 by msukhare          #+#    #+#              #
#    Updated: 2018/09/23 16:14:29 by kemar            ###   ########.fr        #
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
    Y = np.asarray(st.unpack('>'+'B'*nb_labels, label_file.read(nb_labels)),\
            dtype=np.float32).reshape(nb_labels, 1)
    byte_read = (rows * cols * nb_image * 1)
    X = np.asarray(st.unpack('>'+'B'*byte_read, image_file.read(byte_read)),\
            dtype=np.float32).reshape((nb_image, 1, rows, cols))
    return (X, Y, nb_image, rows, cols)

def create_conv_layer(input, filter, p, s, name):
    weights = tf.Variable(tf.truncated_normal(filter, stddev=0.03, seed=5),\
            name=name+"W")
    bias = tf.Variable(tf.truncated_normal([1, filter[3], 24, 24], stddev=0.03, seed=5),\
            name=name+"B")
    conv_layer = tf.nn.conv2d(input, weights, s, padding=p, data_format="NCHW")
    conv_layer += bias
    conv_layer = tf.nn.relu(conv_layer)
    return (conv_layer)

def main():
    if (len(sys.argv) < 4):
        print("need more file")
        sys.exit()
    X_train, Y_train, m, rows, cols = read_file_idx(sys.argv[1], sys.argv[2])
    X_test, Y_test, m, rows, cols = read_file_idx(sys.argv[3], sys.argv[4])
    #plt.imshow(X[45], interpolation='none', cmap='gray')
    #plt.show()
    X_train = X_train / 255
    Y_train = Y_train / 255
    X_test = X_test / 255
    Y_test = Y_test / 255
    tmp = X_train[0:1, :, :, :]
    conv_layer1 = create_conv_layer(tmp, [5, 5, 1, 30], "VALID", [1, 1, 1, 1],\
            "conv_layer1")
    print(conv_layer1)
    conv_layer1 = tf.nn.pool(conv_layer1, [2, 2], pooling_type="MAX",\
            padding="VALID", data_format="NCHW")
    print(conv_layer1)

if __name__ == "__main__":
    main()
