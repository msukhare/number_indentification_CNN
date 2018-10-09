# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    use_modele.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/06 13:11:26 by msukhare          #+#    #+#              #
#    Updated: 2018/10/09 21:37:29 by kemar            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#import tensorflow as tf
import tkinter as tk
import numpy as np
import sys
#from PIL import ImageGrab #===> for macOs
import pyscreenshot as ImageGrab #====> for linux
from PIL import Image
from matplotlib import pyplot as plt
from skimage.transform import resize, pyramid_reduce
import cv2
old_x = None
old_y = None

def get_square(image, square_size):
    height, width = image.shape
   #   differ = height
   # else:
   #   differ = width
   # differ += 4
   # mask = np.zeros((differ, differ), dtype = "uint8")
   # x_pos = int((differ - width) / 2)
   # y_pos = int((differ - height) / 2)
   # mask[y_pos: y_pos + height, x_pos: x_pos + width] = image[0: height, 0: width]
    rows = np.zeros((2, width), dtype=int)
    image = np.r_[rows, image, rows]
    cols = np.zeros(((height + 4), 2), dtype=int)
    image = np.c_[cols, image, cols]
    image = image.astype(np.uint8)
    return (cv2.resize(image, (square_size, square_size), interpolation = cv2.INTER_AREA))

def snap():
    path = "tmp.png"
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    image = ImageGrab.grab((x + 2, y + 2, x + w - 2, y + h - 2))
    image = np.array(image)
    image = image.astype(int)
    image[image == 0] = 255
    image[image == 1] = 0
    return (get_square(image, 28))

def getter():
    img_array = snap()
    plt.imshow(img_array, interpolation='none', cmap='gray')
    plt.show()

def init_button(window):
    button_clear = tk.Button(window, text="clear")
    button_clear.pack()
    button_predict = tk.Button(window, text="predict", command=getter)
    button_predict.pack()
    button_exit = tk.Button(window, text="exit")
    button_exit.pack()

def paint(event):
    global old_x, old_y
    #canvas.create_oval((event.x + 8), (event.y + 3), (event.x - 8), (event.y - 3), fill="black")
    #canvas.create_line((event.x + 4) , (event.y + 2), (event.x - 4), (event.y + 2), fill="black", smooth=tk.TRUE)
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
canvas = tk.Canvas(frame2, width=452, height=452, background='white')
canvas.configure(cursor="crosshair")
canvas.bind("<B1-Motion>", paint)
canvas.bind('<ButtonRelease-1>', reset)
canvas.pack()
window.mainloop()
    #saver = tf.train.Saver()
    #with tf.Session() as sess:
        #saver.restore(sess, "./tmp/model.ckpt")
