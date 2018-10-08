''' Util functions used for building a neural net on MNIST dataset.
'''

from __future__ import print_function
from keras import backend as K

# input image dimensions
img_rows, img_cols = 28, 28

def prepare_images(images):
    if K.image_data_format() == 'channels_first':
        images = images.reshape(images.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        images = images.reshape(images.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    images = images.astype('float32') / 255
    return (images, input_shape)
