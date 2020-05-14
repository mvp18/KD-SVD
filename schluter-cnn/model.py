from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, LeakyReLU, Softmax, Concatenate, Lambda

def Schluter_CNN(dropout_rate=0.0, filter_scale=1):
    ''' Model from Schluter et al. (2015 ISMIR Data Augmentation paper)
    Data input size : (input_frame_size, n_melbins, 1) == (115, 80, 1)
    Args:
        dropout_rate : dropout rate at the dense layer
    '''
    input_shape = (80, 115, 1)
    model = Sequential()
    model.add(Conv2D(64//filter_scale, (3, 3), name='conv1', padding='valid', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(LeakyReLU(0.01))
    model.add(Conv2D(32//filter_scale, (3, 3), name='conv2', padding='valid', kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D((3, 3), strides=(3, 3)))

    model.add(Conv2D(128//filter_scale, (3, 3), name='conv3', padding='valid', kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.01))
    model.add(Conv2D(64//filter_scale, (3, 3), name='conv4', padding='valid', kernel_initializer='he_normal'))
    model.add(LeakyReLU(0.01))
    model.add(MaxPooling2D((3, 3), strides=(3, 3)))

    model.add(Flatten())
    model.add(Dense(256//filter_scale))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64//filter_scale))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2))
    model.add(Softmax(axis=1))

    return model

if __name__ == '__main__':
    model = Schluter_CNN(0.5, 1)
    print(model.summary())
