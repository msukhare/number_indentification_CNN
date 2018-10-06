# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    use_modele.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/06 13:11:26 by msukhare          #+#    #+#              #
#    Updated: 2018/10/06 13:53:54 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf
import sys

def main():
    if (len(sys.argv) < 1):
        print("need more arguments")
        sys.exit()
    with tf.Session() as sess:
        saver.restor(sess, "./tmp/model.ckpt")

if __name__ == "__main__":
    main()
