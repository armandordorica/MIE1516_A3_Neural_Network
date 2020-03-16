from layer import Layer
import numpy as np

from utils import sigmoid_prime


class Activation(Layer):
    def __init__(self, act, act_prime):
        # print("Inside of Activation layer...")
        """
        :param act:             (function) activation function
        :param act_prime:       (function) derivative of the activation function
        """
        # print("Activation Layer created with act: {} and act_prime: {}".format(act, act_prime))
        self.act = act
        self.act_prime = act_prime
        # print("..................................................................\n")

    def forward(self, input_data, mode):
        # print("Inside forward method of Activation Layer ")
        """
        :param input_data:      (numpy.ndarray, shape=[batch_size, # nodes]) output from the previous FCLayer
        :return: output:        (numpy.ndarray, shape=[batch_size, # nodes])
        """
        self.input_data = input_data
        # print("Shape of input_data:{}".format(input_data.shape))

        ########## (E3) Your code goes here ##########
        self.output = self.act(self.input_data)
        output = self.act(self.input_data)
        # print("Shape of activation output (H):{}".format(output.shape))
        ##########            end           ##########

        return output

    def backward(self, delta_n):
        # print("Inside backward method of Activation Layer ")

        """
        Compute and pass the delta to the previous layer
        :param delta_n:         (numpy.ndarray, shape=[batch_size, # nodes]) the delta from the next layer
        :return:                (numpy.ndarray, shape=[batch_size, # nodes]) delta to pass on to the previous layer
        """
        ########## (E3) Your code goes here ##########
        # print("self is of type: {}".format(type(self)))
        # print("Input delta has shape:{}".format(delta_n.shape))
        delta = np.multiply(delta_n, self.act_prime(self.input_data))
        #delta  =  derivative of the loss * sigmoid prime(layer)

        #delta^l = {delta^{l+1}(W^{l+1}}\odot sigmoid_prime(U^l)  # equation 22
        ##########            end           ##########

        return delta

class Dropout(Layer):
    def __init__(self, drop_prob=0.5):
        # print("Inside initialization of Dropout Layer ")
        """
        Dropout is defined as a Layer object for which forward and backward pass should be specified.
        Note that this is so-called 'inverted Dropout' which takes care of scale of outputs in the forward pass.
        :param drop_prob:       (float) probability of dropping a neuron [0, 1]
        """
        super(Dropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, input_data, mode):
        # print("Inside forward method of Dropout Layer")
        """
        Forward pass along the Dropout Layer
        :param input_data:      (numpy.ndarray) output from the previous layer
        :param mode:            (bool)          True during training and False during testing
        :return:                (numpy.ndarray)
        """
        if mode:
            self.mask = (np.random.rand(*input_data.shape) < (1 - self.drop_prob)) / (1 - self.drop_prob)
            return input_data * self.mask
        else:
            return input_data

    def backward(self, delta_n):
        # print("Inside backward method of Dropout Layer")
        return delta_n * self.mask