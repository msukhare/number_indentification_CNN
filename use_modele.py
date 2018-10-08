# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    use_modele.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: msukhare <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2018/10/06 13:11:26 by msukhare          #+#    #+#              #
#    Updated: 2018/10/08 17:15:31 by msukhare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

#import tensorflow as tf
import tkinter as tk
import sys
from PIL import ImageGrab

def snap():
    path = "tmp.png"
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    image = ImageGrab.grab((x + 2, y + 2, x + w - 2, y + h - 2))
    image.save(path)
    return (path)

def getter():
    path = snap()
    

def init_button(window):
    button_clear = tk.Button(window, text="clear")
    button_clear.pack()
    button_predict = tk.Button(window, text="predict", command=getter)
    button_predict.pack()
    button_exit = tk.Button(window, text="exit")
    button_exit.pack()

def paint(event):
    canvas.create_oval((event.x + 8), (event.y + 8), (event.x - 8), (event.y - 8), fill="black")
    canvas.create_line((event.x + 6) , (event.y + 6), (event.x - 6), (event.y - 6), fill="black")

if (len(sys.argv) < 1):
    print("need more arguments")
    sys.exit()
window = tk.Tk()
frame1 = tk.Frame(window)
frame1.pack(side=tk.LEFT, padx=10, pady=10)
init_button(frame1)
frame2 = tk.Frame(window, borderwidth=1, relief=tk.GROOVE)
frame2.pack(side=tk.LEFT, padx=10, pady=10)
canvas = tk.Canvas(frame2, width=280, height=280, background='white')
canvas.configure(cursor="crosshair")
canvas.bind("<B1-Motion>", paint)
canvas.pack()
window.mainloop()
    #saver = tf.train.Saver()
    #with tf.Session() as sess:
        #saver.restore(sess, "./tmp/model.ckpt")
