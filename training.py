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
    '''
    Defines the desired acrhitecture for the neural network model we want to train.

    #Arguments
      input_shape: A shape tuple (integer), not including the batch size. For instance, shape=(32,) indicates that the expected input will be batches of 32-dimensional vectors.
    
    #Returns
       A traineable Keras model.
    '''

    activation_relu = 'relu'
    conv_kernel_size = (3, 3)
    maxpool_filter_size = (2, 2)
    padding_same = 'SAME'

    # First conv layer
    input_tensor = Input(shape=input_shape)
    conv1 = Conv2D(32, kernel_size=conv_kernel_size, activation=activation_relu, padding=padding_same)(input_tensor)
    maxpool2d_1 = MaxPooling2D(maxpool_filter_size)(conv1)

    #Second conv layer
    conv2_1 = Conv2D(64, kernel_size=conv_kernel_size, activation=activation_relu, padding=padding_same)(maxpool2d_1)
    conv2_2 = Conv2D(64, kernel_size=conv_kernel_size, activation=activation_relu, padding=padding_same)(maxpool2d_1)
    maxpool2d_2_1 = MaxPooling2D(maxpool_filter_size)(conv2_1)
    maxpool2d_2_2 = MaxPooling2D(maxpool_filter_size)(conv2_2)

    # Third conv layer
    conv3_1 = Conv2D(256, kernel_size=conv_kernel_size, activation=activation_relu, padding=padding_same)(maxpool2d_2_1)
    conv3_2 = Conv2D(256, kernel_size=conv_kernel_size, activation=activation_relu, padding=padding_same)(maxpool2d_2_2)
    conv3 = concatenate([conv3_1, conv3_2])

    # Flatten the input before passing to fully connected layers.
    flatten = Flatten()(conv3)
    fc1 = Dense(1000, activation=activation_relu)(flatten)
    fc2 = Dense(500, activation=activation_relu)(fc1)
    output_layer = Dense(10, activation='softmax')(fc2)

    model = Model(input_tensor, output_layer)
    model.summary()
    return model

def fit_model(model, x_train, y_train):
    '''
    Returns the model fitted on training data.
    '''
    epochs = 5
    batch_size = 128
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def get_data():
    '''
    The training and test MNIST datasets used to train and evaluate the model.
    '''

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # preprocess images before training.
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

    # save model to disk.
    model.save('mnist_model.h5')

    # evaluate model on training set
    training_loss, training_acc = model.evaluate(x_train, y_train)
    print ('Training loss: {} and accuracy: {}'.format(training_loss, training_acc))

    # evaluate model on test set
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print ('Test loss: {} and accuracy: {}'.format(test_loss, test_acc))
