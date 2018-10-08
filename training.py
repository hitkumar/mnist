'''
  Trains a neural network model on MNIST dataset and saves the model to disk.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.utils import to_categorical
from keras import backend as K
from keras import Input
from keras.layers import Conv2D, MaxPooling2D, concatenate, Flatten, Dense
from keras.utils import plot_model
from utils import prepare_images

def get_model_architecture(input_shape):
    activation_relu = 'relu'
    conv_kernel_size = (3, 3)
    maxpool_filter_size = (2, 2)

    input_tensor = Input(shape=input_shape)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='SAME')(input_tensor)
    maxpool2d_1 = MaxPooling2D((2, 2))(conv1)
    conv2_1 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='SAME')(maxpool2d_1)
    conv2_2 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='SAME')(maxpool2d_1)

    maxpool2d_2_1 = MaxPooling2D((2, 2))(conv2_1)
    maxpool2d_2_2 = MaxPooling2D((2, 2))(conv2_2)
    conv3_1 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='SAME')(maxpool2d_2_1)
    conv3_2 = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='SAME')(maxpool2d_2_2)
    conv3 = concatenate([conv3_1, conv3_2])

    flatten = Flatten()(conv3)
    fc1 = Dense(1000, activation='relu')(flatten)
    fc2 = Dense(500, activation='relu')(fc1)
    output_layer = Dense(10, activation='softmax')(fc2)

    model = Model(input_tensor, output_layer)
    model.summary()
    return model

def fit_model(model, x_train, y_train):
    epochs = 1
    batch_size = 128
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def get_data():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, train_input_shape = prepare_images(x_train)
    x_test, test_input_shape = prepare_images(x_test)
    assert train_input_shape == test_input_shape

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (x_train, y_train, x_test, y_test, train_input_shape)

if __name__ == "__main__":
    # the data, split between train and test sets
    (x_train, y_train, x_test, y_test, input_shape) = get_data()
    model = fit_model(get_model_architecture(input_shape), x_train, y_train)

    model.save('mnist_model.h5')
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print (test_loss, test_acc)
