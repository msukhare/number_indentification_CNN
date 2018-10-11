# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    use_modele.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/06 13:11:26 by msukhare          #+#    #+#              #
#    Updated: 2018/10/11 16:49:36 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import tkinter as tk
import numpy as np
import tensorflow as tf
import cv2
import sys
from PIL import ImageGrab #===> for macOs
#import pyscreenshot as ImageGrab #====> for linux
from PIL import Image
from matplotlib import pyplot as plt

old_x = None
old_y = None

def resize_image(image):
    height, width = image.shape[0], image.shape[1]
    rows = np.zeros((2, width), dtype=np.uint8)
    image = np.r_[rows, image, rows]
    cols = np.zeros(((height + 4), 2), dtype=np.uint8)
    image = np.c_[cols, image, cols]
    return (cv2.resize(image, (28, 28), interpolation = cv2.INTER_AREA))

def add_padding(img, pad_l, pad_t, pad_r, pad_b):
    height, width = img.shape
    pad_left = np.zeros((height, pad_l), dtype = np.int)
    img = np.concatenate((pad_left, img), axis = 1)
    pad_up = np.zeros((pad_t, pad_l + width))
    img = np.concatenate((pad_up, img), axis = 0)
    pad_right = np.zeros((height + pad_t, pad_r))
    img = np.concatenate((img, pad_right), axis = 1)
    pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
    img = np.concatenate((img, pad_bottom), axis = 0)
    return (img)

def center_number(image):
    col_sum = np.where(np.sum(image, axis=0) > 0)
    row_sum = np.where(np.sum(image, axis=1) > 0)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
    image = image[y1: y2, x1: x2]
    plt.imshow(image, interpolation='none', cmap='gray')
    plt.show()
    return (add_padding(image, 40, 40, 40, 40))

def snap():
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    image = np.array(ImageGrab.grab((x + 2, y + 2, x + w - 2, y + h - 2)).convert('1'))
    image = image.astype(np.uint8)
    image[image == 0] = 255
    image[image == 1] = 0
    return (resize_image(center_number(image)))

def pred():
    img_array = snap()
    plt.imshow(img_array, interpolation='none', cmap='gray')
    plt.show()
    img_array = np.reshape(img_array, (1, 28, 28, 1))
    #Y = np.zeros((1, 10), dtype=int)
    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph('./tmp/my_model.ckpt.meta')
        saver.restore(sess, './tmp/my_model.ckpt')
        out_put = sess.run('out:0', feed_dict={'x:0': img_array})
        print(out_put)
    img_array = np.reshape(img_array, (28, 28))

def init_button(window):
    button_clear = tk.Button(window, text="clear")
    button_clear.pack()
    button_predict = tk.Button(window, text="predict", command=pred)
    button_predict.pack()
    button_exit = tk.Button(window, text="exit")
    button_exit.pack()

def paint(event):
    global old_x, old_y
    if (old_x and old_y):
        canvas.create_line(old_x, old_y, event.x, event.y, width=30, fill="black", capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
    old_x = event.x
    old_y = event.y

def reset(event):
    global old_x, old_y
    old_x, old_y = None, None

if (len(sys.argv) < 1):
    print("need more arguments")
    sys.exit()
window = tk.Tk()
frame1 = tk.Frame(window)
frame1.pack(side=tk.LEFT, padx=10, pady=10)
init_button(frame1)
frame2 = tk.Frame(window, borderwidth=1, relief=tk.GROOVE)
frame2.pack(side=tk.LEFT, padx=10, pady=10)
canvas = tk.Canvas(frame2, width=448, height=448, background='white')
canvas.configure(cursor="crosshair")
canvas.bind("<B1-Motion>", paint)
canvas.bind('<ButtonRelease-1>', reset)
canvas.pack()
window.mainloop()
