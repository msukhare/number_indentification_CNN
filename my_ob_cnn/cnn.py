# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    cnn.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/09/18 09:48:07 by msukhare          #+#    #+#              #
#    Updated: 2018/09/18 15:49:38 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys

class cnn:

    def __init__(self):
        #to make architecture call add_arc(typelayer, )
        #type_layer = convop--> 1, poolmax-> 2, poolaverage-->3, drop_out-->4, flattent-->5,
        #fullyconnected->6, out_put->7
        self.architecture = []
        #conv op
        self.activ_funct = []
        self.filter_size = []
        self.s = []
        self.p = []
        self.nb_channels = [1]
        self.nb_filter = []
        self.filters = []
        #pooling
        self.type_pooling = []
        self.size_pooling = []
        #drop
        self.drops = []
        #fully_connected_layer/hiden_layers
        self.nb_neurones = []
        self.act_fun_layer = []

    def convlayer(self, f, s, nb_filter, p, activate_func, type, nb_channels):
        self.architecture.apprend(1)
        if (f <= 1):
            print("size filter must be >= 2, will be set at 2*2 by default")
            f = 2
        if (s <= 0):
            print("stride must be >= 1, will be set at 1 by default")
            s = 1
        if (p <= 0):
            print("padding must be >= 1, will be set at 1 by default")
            p = 1
        if (nb_filter <= 0):
            print("nb_filter must be >= 1, will be set at 1 by default")
            nb_filter = 1
        if (activate_func != 1 or activate_func != 2):
            print("activate_func must be 1 or 2, will be set at 1 by default")
            activ_funct = 1
        self.nb_filter.append(nb_filter)
        self.s.append(s)
        self.p.append(p)
        self.activ_funct.append(activate_func)
        self.filter_size.append(f)

    def pool(sefl, type, size_pooling):
        self.architecture.apprend(2)
        if (type != 1 or type != 2):
            print("type of pooling must be 1 or 2, will be set at 1 by default")
            type = 1
        if (size_pooling <= 1):
            print("size of pooling must be >= 2, will be set at 2 by default")
            size_pooling = 2
        self.type_pooling.append(type)
        self.size_pooling.append(size_pooling)

    def drop_out(self, size_drop):
        self.architecture.apprend(3)
        if (size_drop <= 0 or size_drop > 99):
            print("drop must be >= 1 and < 99, will be set 10%")
            size_drop = 10
        self.drops.append(size_drop)

    def flattent(self):
        self.architecture.append(4)

    def fully_connected(self, nb_neurones, activation_func):
        if (nb_neurones <= 0):
            print("nb_neurones must be >= 1, will be set at 1 by default")
            nb_neurones = 1
        self.nb_neurones.append(nb_neurones)
        if (activation_func == 3):
            self.architecture.append(6)
            self.act_fun_layer.append(activation_func)
            return
        if (activation_func != 1 or activation_func != 2):
            print("activate_func must be 1 or 2 or 3, will be set at 1 by default")
            activation_func = 1
        self.architecture.append(5)
        self.act_fun_layer.append(activation_func)

    def init_filters(self):
        return 

