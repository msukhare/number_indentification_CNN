# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    use_model.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/12 10:56:26 by msukhare          #+#    #+#              #
#    Updated: 2018/11/11 16:48:43 by kemar            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tkinter as tk
import numpy as np
import tensorflow as tf
import cv2
import sys
from matplotlib import pyplot as plt
from PIL import ImageGrab #===> for macOs
#import pyscreenshot as ImageGrab #====> for linux
from PIL import Image

class digit_recognition:

    def __init__(self):

        #Init tkinter
        self.old_x = None
        self.old_y = None
        self.window = tk.Tk()
        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.init_button()
        self.draw_frame = tk.Frame(self.window, borderwidth=1, relief=tk.GROOVE)
        self.draw_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.draw_can = tk.Canvas(self.draw_frame, width=448, height=448, background='white')
        self.draw_can.configure(cursor="crosshair")
        self.draw_can.bind("<B1-Motion>", self.paint)
        self.draw_can.bind('<ButtonRelease-1>', self.reset)
        self.draw_can.pack()
        self.pred_frame = tk.Frame(self.draw_frame, borderwidth=1, relief=tk.GROOVE)
        self.pred_frame.pack(side=tk.BOTTOM, padx=0, pady=0)
        self.textlabel = tk.Label(self.pred_frame, text="prediction by the model:", bg="white")
        self.textlabel.pack()

        # Init tensorflow
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph('./tmp/my_model.ckpt.meta')
        self.saver.restore(self.sess, './tmp/my_model.ckpt')

    def center_image(self, image, padd, size_max, side_to_add, axis):
        padd_modif = padd
        size_to_add = size_max - side_to_add
        if ((size_to_add % 2) != 0):
            if ((int(size_to_add / 2) * 2 + side_to_add) > size_max):
                padd_modif -= (int(size_to_add / 2) * 2 + side_to_add) - size_max
            elif ((int(size_to_add / 2) * 2 + side_to_add) < size_max):
                padd_modif += size_max - (int(size_to_add / 2) * 2 + side_to_add)
        if (axis == 1):
            cols_1 = np.zeros((size_max, (int(size_to_add / 2) + padd)), dtype=np.uint8)
            cols_2 = np.zeros((size_max, (int(size_to_add / 2) + padd_modif)), dtype=np.uint8)
            image = np.c_[cols_1, image, cols_2]
            rows = np.zeros((padd, image.shape[1]), dtype=np.uint8)
            return (np.r_[rows, image, rows])
        rows_1 = np.zeros(((int(size_to_add / 2) + padd), size_max), dtype=np.uint8)
        rows_2 = np.zeros(((int(size_to_add / 2) + padd_modif), size_max), dtype=np.uint8)
        image = np.r_[rows_1, image, rows_2]
        cols = np.zeros((image.shape[0], padd), dtype=np.uint8)
        return (np.c_[cols, image, cols])

    def resize_image(self, image, padd):
        y, x = np.shape(image)
        if (y > x):
            image = self.center_image(image, padd, y, x, 1)
        elif (x > y):
            image = self.center_image(image, padd, x, y, 0)
        else:
            cols = np.zeros((y, padd), dtype=np.uint8)
            image = np.c_[cols, image, cols]
            rows = np.zeros((padd, (x + 2 * padd)), dtype=np.uint8)
            image = np.r_[rows, image, rows]
        #print(np.shape(image))
        #plt.imshow(image, interpolation='none', cmap='gray')
        #plt.show()
        return (cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA))

    def cut_number_in_image(self, image):
        if (np.amax(image) == 0):
            return (image)
        col_sum = np.where(np.sum(image, axis=0) > 0)
        row_sum = np.where(np.sum(image, axis=1) > 0)
        y1, y2 = row_sum[0][0], row_sum[0][-1]
        x1, x2 = col_sum[0][0], col_sum[0][-1]
        image = image[y1: y2, x1: x2]
        #plt.imshow(image, interpolation='none', cmap='gray')
        #plt.show()
        return (image)

    def snap(self):
        x = self.draw_can.winfo_rootx()
        y = self.draw_can.winfo_rooty()
        w = self.draw_can.winfo_width()
        h = self.draw_can.winfo_height()
        image = np.array(ImageGrab.grab((x + 2, y + 2, x + w - 2, y + h - 2)).convert('1'))
        image = image.astype(np.uint8)
        image[image == 0] = 255
        image[image == 1] = 0
        return (self.resize_image(self.cut_number_in_image(image), 100))

    def print_pred(self, out_put):
        class_pred = np.where(out_put == np.amax(out_put))
        self.textlabel.pack_forget()
        self.textlabel = tk.Label(self.pred_frame,\
                text="prediction by the model: {0}".format(class_pred[1]), bg="white")
        self.textlabel.pack()

    def pred(self):
        img_array = self.snap()
        #plt.imshow(img_array, interpolation='none', cmap='gray')
        #plt.show()
        img_array = np.reshape(img_array, (1, 28, 28, 1))
        self.print_pred(self.sess.run('out:0', feed_dict={'x:0': img_array}))

    def clear(self):
        self.textlabel.pack_forget()
        self.textlabel = tk.Label(self.pred_frame, text="prediction by the model:", bg="white")
        self.textlabel.pack()
        self.draw_can.pack_forget()
        self.draw_can = tk.Canvas(self.draw_frame, width=448, height=448, background='white')
        self.draw_can.configure(cursor="crosshair")
        self.draw_can.bind("<B1-Motion>", self.paint)
        self.draw_can.bind('<ButtonRelease-1>', self.reset)
        self.draw_can.pack()

    def init_button(self):
        button_clear = tk.Button(self.button_frame, text="clear", command=self.clear)
        button_clear.pack()
        button_predict = tk.Button(self.button_frame, text="predict", command=self.pred)
        button_predict.pack()
        button_exit = tk.Button(self.button_frame, text="exit", command=self.window.quit)
        button_exit.pack()

    def paint(self, event):
        if (self.old_x and self.old_y):
            self.draw_can.create_line(self.old_x, self.old_y, event.x, event.y, width=30,\
                    fill="black", capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=tk.TRUE,\
                    splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

def main():
    digit_reco = digit_recognition()
    digit_reco.window.mainloop()
    digit_reco.sess.close()

if __name__ == "__main__":
    main()
