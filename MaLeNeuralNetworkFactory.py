from MaLeConvSeq import MaLeConvSeq


class MaLeNeuralNetworkFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_network(n_type,  xshape, neuron):
        __network_classes = {
            'MaLeConvSeq': MaLeConvSeq
        }
        return __network_classes[n_type](xshape, neuron)
