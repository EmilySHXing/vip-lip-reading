import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.callbacks import TensorBoard
import pickle
import time
import numpy as np

# 0: "About"     1: "Absolutely"    2: "Abuse"      3: "Access"    4: "According"   5: "Accused"
# 6: "Across"    7: "Action"        8: "Actually"   9: "Allow"     10: "Today"      11: "Tomorrow"    12: "Yesterday"

NAME = "13-Words-8-16-16-32-d256-{}".format(int(time.time()))

tensor_b = TensorBoard(log_dir='logs/{}'.format(NAME))

# load data
x_train1 = pickle.load(open("data/data_train1.pickle", "rb"))
x_train2 = pickle.load(open("data/data_train2.pickle", "rb"))
x_train = np.concatenate((x_train1, x_train2))
y_train = pickle.load(open("data/label_train.pickle", "rb"))
x_test = pickle.load(open("data/data_test1.pickle", "rb"))
y_test = pickle.load(open("data/label_test.pickle", "rb"))
x_val = pickle.load(open("data/data_val1.pickle", "rb"))
y_val = pickle.load(open("data/label_val.pickle", "rb"))

# normalize to 0-1
x_train = x_train/255.0
x_test = x_test/255.0
x_val = x_val/255.0

# initializations
kernel_size = (3,3,3)
filters_1 = 8
filters_2 = 16
filters_3 = 16
filters_4 = 32

# create model
model = tf.keras.Sequential()
# one convolution layer with 4 filters of size 3x3x3, followed by relu activation
model.add(layers.Conv3D(filters_1, kernel_size, strides=(1, 1, 1), padding='same', input_shape=(29,32,32,1),
                        dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='TruncatedNormal',
                        bias_initializer='zeros'))
# max pooling, window 2x2x2, stride 2x2x2
model.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2)))
# second convolution layer with 8 filters of size 3x3x3, followed by relu activation
model.add(layers.Conv3D(filters_2, kernel_size, strides=(1, 1, 1), padding='same',
                        dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='TruncatedNormal',
                        bias_initializer='zeros'))
# max pooling, window 2x2x2, stride 2x2x2
model.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2)))
# third convolution layer with 8 filters of size 3x3x3, followed by relu activation
model.add(layers.Conv3D(filters_3, kernel_size, strides=(1, 1, 1), padding='same',
                        dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='TruncatedNormal',
                        bias_initializer='zeros'))
# fourth convolution layer with 8 filters of size 3x3x3, followed by relu activation
model.add(layers.Conv3D(filters_4, kernel_size, strides=(1, 1, 1), padding='same',
                        dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='TruncatedNormal',
                        bias_initializer='zeros'))
# max pooling, window 2x2x2, stride 2x2x2
model.add(layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2)))
# flatten into single vector
model.add(layers.Flatten())
# dense layer, output vector of length 256
model.add(layers.Dense(256, activation="relu"))
# dropout layer with rate 0.5
model.add(layers.Dropout(0.5))
# fully connected layer, out put a vector of length 6
model.add(layers.Dense(13, activation="softmax"))

# compile using cross-entropy loss and adam optimizer
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# train model with training set
model.fit(x_train, y_train, batch_size=128, validation_data=(x_val, y_val), epochs=10, callbacks=[tensor_b])

# print model structure summary
model.summary()

# evaluate model with test set
model.evaluate(x_test, y_test)

# save model
model.save("models/model-8-16-16-32-d256")