# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    cnn_architecture.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/04 09:42:59 by msukhare          #+#    #+#              #
#    Updated: 2018/10/10 17:19:32 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf

def define_dense_layers(in_put, nb_before, nb_neurones):
    weights = tf.Variable(tf.truncated_normal([nb_before, nb_neurones],\
            stddev=0.05, seed=5), name="weights1")
    bias = tf.Variable(tf.truncated_normal([1, nb_neurones], stddev=0.05, seed=5),\
            name="bias1")
    return (tf.nn.relu(tf.matmul(in_put, weights) + bias))

def create_conv_layer(input, filter, p, s, bias, name, act_funct):
    weights = tf.Variable(tf.truncated_normal(filter, stddev=0.05, seed=5),\
            name=name+"W")
    bias = tf.Variable(tf.truncated_normal(bias, stddev=0.05, seed=5),\
            name=name+"B")
    conv_layer = tf.nn.conv2d(input, weights, s, padding=p)
    conv_layer += bias
    if (act_funct == 1):
        conv_layer = tf.nn.relu(conv_layer)
    return (conv_layer)

def define_conv_for_net5(x):
    layer1 = create_conv_layer(x, [5, 5, 1, 6], "VALID", [1, 1, 1, 1],\
            [1, 1, 6], "first_conv_layer", 1)
    layer1 = tf.nn.pool(layer1, [2, 2], pooling_type="MAX",\
            padding="VALID", strides=[2, 2])
    layer2 = create_conv_layer(layer1, [5, 5, 6, 16], "VALID", [1, 1, 1, 1],\
            [1, 1, 16], "sec_conv_layer", 1)
    layer2 = tf.nn.pool(layer2, [2, 2], pooling_type="MAX",\
            padding="VALID", strides=[2, 2])
    return (layer2)

def init_net5(x, y):
    out_conv = define_conv_for_net5(x)
    flatten_layer = tf.reshape(out_conv, [-1, (out_conv.shape[1] *\
            out_conv.shape[2] * out_conv.shape[3])])
    out_dense = define_dense_layers(flatten_layer, 256, 120)
    out_dense = define_dense_layers(out_dense, 120, 84)
    out_dense = define_dense_layers(out_dense, 84, 10)
    return (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(\
            logits=out_dense, labels=y, name="cross")), tf.nn.softmax(out_dense, name="out"))

def define_conv_for_random_cnn(x):
    layer1 = create_conv_layer(x, [5, 5, 1, 30], "VALID", [1, 1, 1, 1],\
            [1, 1, 30], "first_conv_layer", 1)
    layer1 = tf.nn.pool(layer1, [2, 2], pooling_type="MAX",\
            padding="VALID", strides=[2,2])
    layer2 = create_conv_layer(layer1, [3, 3, 30, 15], "VALID", [1, 1, 1, 1],\
            [1, 1, 15], "sec_conv_layer", 1)
    layer2 = tf.nn.pool(layer2, [2, 2], pooling_type="MAX",\
            padding="VALID", strides=[2, 2])
    return (layer2)

def init_a_random_cnn(x, y):
    out_conv = define_conv_for_random_cnn(x)
    drop = tf.nn.dropout(out_conv, 0.2)
    flatten_layer = tf.reshape(drop, [-1, (drop.shape[1] *\
            drop.shape[2] * drop.shape[3])])
    out_dense = define_dense_layers(flatten_layer, 375, 128)
    out_dense = define_dense_layers(out_dense, 128, 50)
    out_dense = define_dense_layers(out_dense, 50, 10)
    return (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(\
            logits=out_dense, labels=y, name="cross")),\
            tf.nn.softmax(out_dense, name="out"))
