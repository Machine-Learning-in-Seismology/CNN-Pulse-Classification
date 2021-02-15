import keras
import keras_metrics
from keras import Sequential
from keras.layers import Conv1D, Flatten, Dense, Reshape, MaxPooling1D, Dropout

from MaLeNeuralNetwork import MaLeNeuralNetwork


class MaLeConvNet(MaLeNeuralNetwork):
    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, val):
        self.__model = val

    def __init__(self, xshape, neuron):
        super(MaLeNeuralNetwork, self).__init__()
        self.model = Sequential()
        self.model.add(Reshape((xshape[1], 1), input_shape=(xshape[1],)))
        self.model.add(Conv1D(16, kernel_size=12, activation='relu', input_shape=(xshape[1], 1),
                              kernel_initializer='glorot_normal'))
        self.model.add(MaxPooling1D(pool_size=4))
        self.model.add(Conv1D(16, kernel_size=6, activation='relu', kernel_initializer='glorot_normal'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_normal'))
        self.model.add(MaxPooling1D(pool_size=3))
        self.model.add(Conv1D(64, kernel_size=3, activation='relu', kernel_initializer='glorot_normal'))
        self.model.add(MaxPooling1D(pool_size=3))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(20, activation='relu', kernel_initializer='glorot_normal'))
        self.model.add(Dense(10, activation='relu', kernel_initializer='glorot_normal'))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=[keras_metrics.false_positive(), keras_metrics.true_negative(),
                                    keras_metrics.false_negative(), keras_metrics.true_positive(),
                                    keras.metrics.binary_accuracy])

    # def train(self, x, y, epoch):
    #     x = x.reshape(x.shape[0], x.shape[1], 1)
    #     super(MaLeConvNet, self).train(x, y, epoch)
    #
    # def evaluate(self, x, y):
    #     x = x.reshape(x.shape[0], x.shape[1], 1)
    #     return super(MaLeConvNet, self).evaluate(x, y)
