# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_cnn.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/17 16:28:47 by msukhare          #+#    #+#              #
#    Updated: 2018/09/18 16:49:45 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import struct as st
import sys
from matplotlib import pyplot as plt
import tensorflow as tf

def read_file_idx(images, labels):
    image_file = open(images, "rb")
    label_file = open(labels, "rb")
    magic_image, nb_image, rows, cols = st.unpack('>IIII',image_file.read(16))
    magic_label, nb_labels = st.unpack('>II', label_file.read(8))
    Y = np.asarray(st.unpack('>'+'B'*nb_labels, label_file.read(nb_labels))).reshape(nb_labels, 1)
    byte_read = (rows * cols * nb_image * 1)
    X = np.asarray(st.unpack('>'+'B'*byte_read, image_file.read(byte_read))).reshape((nb_image,\
            rows, cols))
    return (X, Y, nb_image, rows, cols)

def main():
    X_train, Y_train, m, rows, cols = read_file_idx("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    X_test, Y_test, m, rows, cols = read_file_idx("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
    #plt.imshow(X[45], interpolation='none', cmap='gray')
    #plt.show()
    X_train = X_train / 255
    Y_train = Y_train / 255
    X_test = X_test / 255
    Y_test = Y_test / 255



if __name__ == "__main__":
    main()
