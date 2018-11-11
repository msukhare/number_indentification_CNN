# number_indentification_CNN

## About number_indentification_CNN:

* It's a machine learning project.

* This project uses [tensorflow](https://www.tensorflow.org/) to do an [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network).

* number_identification_CNN is composed of two scripts, `train_cnn.py` and `use_model.py`.

## About `train_cnn.py`:

* `train_cnn.py` uses [tensorflow](https://www.tensorflow.org/) to create an CNN, train CNN and save the model in tmp.

* `train_cnn.py` calls init_net5 in cnn_architecture to create [LeNet5](http://yann.lecun.com/exdb/lenet/).

* `train_cnn.py` uses gradientDescenteOptimizer with mini batch to train weights.

* The cost function is a cross entropy.

* This script uses a softmax function in output layer.

* You can also use AdamOptimizer if you remove comment on line 82, and put a comment on line 81.

* Another architecture is available in `train_cnn.py`, it's a random CNN find on a blog. You can remove comment on line 190, and put a comment on line 191.

## About `use_cnn.py`:

* `use_cnn.py` creates a window with a canvas where you can draw number like in paint:

![main_page](exemple/main_page)

* draw here:

![draw_canvas](exemple/draw)

* To predict number use predict buttom:

![buttom_predict](exemple/buttom_predict)

* To clear use clear buttom:

![buttom_clear](exemple/clear)

* To exit use exit buttom or kill script:

![exit_buttom](exemple/exit_buttom)

* When you use predict buttom, the prediction is show down draw canvas like this:

!prediction](exemple/prediction)

* After make a prediction you can clear or use predict buttom for an new prediction.

* This script uses [tkinter](https://en.wikipedia.org/wiki/Tkinter) to generate a window, canvas and buttom.

* This scipt uses tensorflow to restaur model in tmp.

* It's use to [openCV](https://opencv.org/) to resize image in (28, 28).

* _**If you are on linux distribution remove comment line 20, and comment line 19, because pysceenshot doesn't work on macOS.**_

### About data train and test:

