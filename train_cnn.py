# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train_cnn.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/17 16:28:47 by msukhare          #+#    #+#              #
#    Updated: 2018/09/17 17:01:58 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import struct as st
import sys
from matplotlib import pyplot as plt

def read_file_idx():
    image_file = open("train-images-idx3-ubyte", "rb")
    label_file = open("train-labels-idx1-ubyte", "rb")
    magic_image, nb_image, rows, cols = st.unpack('>IIII',image_file.read(16))
    magic_label, nb_labels = st.unpack('>II', label_file.read(8))
    Y = np.asarray(st.unpack('>'+'B'*nb_labels, label_file.read(nb_labels))).reshape(nb_labels, 1)
    byte_read = (rows * cols * nb_image * 1)
    X = np.asarray(st.unpack('>'+'B'*byte_read, image_file.read(byte_read))).reshape((nb_image,\
            rows, cols))
    return (X, Y, nb_image, rows, cols)


def main():
    X, Y, m, rows, cols = read_file_idx()
    plt.imshow(X[45], interpolation='none')
    plt.show()


if __name__ == "__main__":
    main()
