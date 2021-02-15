import keras
import keras_metrics
from keras import Input, Model
from keras.layers import Conv1D, Flatten, Dense, Reshape, MaxPooling1D, Dropout, concatenate, Activation
import numpy as np
from sklearn.preprocessing import StandardScaler

from MaLeNeuralNetwork import MaLeNeuralNetwork


class MaLeConvSeq(MaLeNeuralNetwork):
    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, val):
        self.__model = val

    def __init__(self, xshape, neuron):
        super(MaLeNeuralNetwork, self).__init__()
        inputs = Input(shape=(xshape[1],))
        inminmax = Input(shape=(2,))
        inre = Reshape((xshape[1], 1), input_shape=(xshape[1],))(inputs)
        
        conv1 = Conv1D(16, kernel_size=12, activation='relu', kernel_initializer='glorot_normal')(inre)
        pool1 = MaxPooling1D(pool_size=4)(conv1)

        conv2 = Conv1D(16, kernel_size=6, activation='relu', kernel_initializer='glorot_normal')(pool1)
        pool2 = MaxPooling1D(pool_size=2)(conv2)

        conv3 = Conv1D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(pool2)
        pool3 = MaxPooling1D(pool_size=3)(conv3)

        conv4 = Conv1D(64, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(pool3)
        pool4 = MaxPooling1D(pool_size=3)(conv4)

        dropout = Dropout(0.5)(pool4)

        flat = Flatten(name='flatten')(dropout)
        concat = concatenate([flat, inminmax], axis=1)

        den1 = Dense(20, activation='relu', kernel_initializer='glorot_normal')(concat)
        den2 = Dense(10, activation='relu', kernel_initializer='glorot_normal')(den1)
        # den3 = Dense(30, activation='relu', kernel_initializer='glorot_normal')(den2)
        den4 = Dense(1)(den2)
        pred = Activation('sigmoid', name='sigmoid')(den4)

        self.model = Model(inputs=[inputs, inminmax], outputs=pred)
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=[keras_metrics.false_positive(), keras_metrics.true_negative(),
                                    keras_metrics.false_negative(), keras_metrics.true_positive(),
                                    keras.metrics.binary_accuracy])

    def input_scale(self, x_train, x_test):
        min = np.min(x_train, axis=1)
        max = np.max(x_train, axis=1)
        trainmm = np.array((min, max)).T
        min = np.min(x_test, axis=1)
        max = np.max(x_test, axis=1)
        testmm = np.array((min, max)).T
        scaler = StandardScaler()
        scaler = scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        return [x_train, trainmm], [x_test, testmm]