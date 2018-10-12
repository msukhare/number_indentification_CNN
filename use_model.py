# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    use_model.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/12 10:56:26 by msukhare          #+#    #+#              #
#    Updated: 2018/10/12 17:04:36 by msukhare         ###   ########.fr        #
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

    def resize_image(self, image):
        height, width = image.shape[0], image.shape[1]
        rows = np.zeros((2, width), dtype=np.uint8)
        image = np.r_[rows, image, rows]
        cols = np.zeros(((height + 4), 2), dtype=np.uint8)
        image = np.c_[cols, image, cols]
        return (cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA))

    def add_padding(self, img, padd):
        y, x = np.shape(img)
        if (y > x):
            cols = np.zeros((y, int((y - x) / 2)), dtype=np.uint8)
            img = np.c_[cols, img, cols]
        else:
            rows = np.zeros((int((x - y) / 2), x), dtype=np.uint8)
            img = np.r_[rows, img, rows]
        y, x = np.shape(img)
        cols = np.zeros((y, padd), dtype=np.uint8)
        img = np.c_[cols, img, cols]
        rows = np.zeros((padd, (x + 2* padd)), dtype=np.uint8)
        img = np.r_[rows, img, rows]
        plt.imshow(img, interpolation='none', cmap='gray')
        plt.show()
        print(np.shape(img))
        return (img)

    def center_number(self, image):
        col_sum = np.where(np.sum(image, axis=0) > 0)
        row_sum = np.where(np.sum(image, axis=1) > 0)
        y1, y2 = row_sum[0][0], row_sum[0][-1]
        x1, x2 = col_sum[0][0], col_sum[0][-1]
        image = image[y1: y2, x1: x2]
        plt.imshow(image, interpolation='none', cmap='gray')
        plt.show()
        return (self.add_padding(image, 50))

    def snap(self):
        x = self.draw_can.winfo_rootx()
        y = self.draw_can.winfo_rooty()
        w = self.draw_can.winfo_width()
        h = self.draw_can.winfo_height()
        image = np.array(ImageGrab.grab((x + 2, y + 2, x + w - 2, y + h - 2)).convert('1'))
        image = image.astype(np.uint8)
        image[image == 0] = 255
        image[image == 1] = 0
        return (self.resize_image(self.center_number(image)))

    def print_pred(self, out_put):
        class_pred = np.where(out_put == np.amax(out_put))
        self.textlabel.pack_forget()
        self.textlabel = tk.Label(self.pred_frame,\
                text="prediction by the model: {0}".format(class_pred[1]), bg="white")
        self.textlabel.pack()

    def pred(self):
        img_array = self.snap()
        plt.imshow(img_array, interpolation='none', cmap='gray')
        plt.show()
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
