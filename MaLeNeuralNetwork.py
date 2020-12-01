import abc
import numpy as np

from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from keras.utils.vis_utils import plot_model


class MaLeNeuralNetwork(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def model(self):
        pass

    @model.setter
    def model(self, val):
        pass

    def train(self, x, y, epoch):
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto', restore_best_weights=True)
        self.model.fit(x=x, y=y, batch_size=1000, epochs=epoch, validation_split=0.25,
                       callbacks=[early_stop])
        return early_stop.stopped_epoch

    def evaluate(self, x, y):
        score = self.model.evaluate(x, y)
        return score

    def input_scale(self, x_train, x_test):
        return x_train, x_test

    def predict(self, x):
        return self.model.predict(x)

    def print(self, x):
        print(self.model.summary())
