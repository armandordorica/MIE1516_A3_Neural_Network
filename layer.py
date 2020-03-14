import numpy as np
import math
import scipy.stats as stats
from param import Parameter
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, input_data, mode):
        # print("Inside abstract forward method of Layer")
        raise NotImplementedError

    @abstractmethod
    def backward(self, delta_n):
        # print("Inside abstract backward method of Layer")
        """
        :param delta_n: the delta from the next layer
        """
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, input_size, output_size, initialization='normal', uniform=False):
        # print("Inside initialization method of FCLayer")
        # print("Input size:{}".format(input_size))
        # print("Output size:{}".format(output_size))
        """
        :param input_size:      (int)   the number of nodes in the previous layer
        :param output_size:     (int)   the number of nodes in 'this' layer
        :param initialization:  (str)   whether to use Xavier initialization
        :param uniform:         (bool)  in Xavier initialization, whether to use uniform or truncated normal distribution
        """
        if initialization == 'xavier':
            # print("Chosen initialization is Xavier")
            # Xavier initialization followed the implementation in Tensorflow.
            fan_in = input_size
            fan_out = output_size
            n = (fan_in + fan_out) / 2.0
            if uniform:
                limit = math.sqrt(3.0 / n)
                weights = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
                bias = np.zeros((1, output_size))
            else:
                trunc_std = math.sqrt(1.3 / n)
                a, b = -2, 2                            # truncated between [- 2 * std, 2 * std]
                weights = stats.truncnorm.rvs(a, b, loc=0.0, scale=trunc_std, size=(fan_in, fan_out))
                bias = np.zeros((1, output_size))
        else:
            weights = np.random.randn(input_size, output_size)
            bias = np.random.randn(1, output_size)



        self.weights = Parameter(weights)               # instantiate Parameter object by passing the initialized values
        self.bias = Parameter(bias)                     # instantiate Parameter object by passing the initialized values
        self.param = [self.weights, self.bias]

        # print("Weights are: {}".format(self.weights.value))
        # print("Shape of weights matrix is: {}".format(self.weights.value.shape))
        # print("Bias matrix is: {}".format(self.bias.value))
        # print("Shape of Bias matrix is: {}".format(self.bias.value.shape))
        # print("self.param: {}".format(self.param))
        # print("Len of self.param list is: {}".format(len(self.param)))
        #
        # print("FC Layer created")
        # print("..................................................................\n")
        self.backward_counter = 0


        # store the weight and bias to a list

    def forward(self, input_data, mode):
        # print("Inside forward propagation of FCLayer ")
        """
        If self.weights.shape = (in, out), then
        :param input_data:      (numpy.ndarray, shape=[batch_size, in]) the output from the previous layer
        :return: output         (numpy.ndarray, shape=[batch_size, out])
        """
        n = input_data.shape[0]
        # print("Size of mini batch : {}".format(n))# size of mini-batch
        self.input_data = input_data        # store the input as attribute (to use in backpropagation)

        # print("Input data:{}".format(input_data))
        # print("self.weights: {}".format(self.weights))
        # print("self.bias:{}".format(self.bias))

        # print("Input data shape :{}".format(input_data.shape))
        # print("self.weights shape: {}".format(self.weights.shape))
        # print("self.bias shape:{} ".format(self.bias.shape))

        ########## (E2) Your code goes here ##########
        # print("shape of np.dot(self.input_data, self.weights.value): {}".format(np.dot(self.input_data, self.weights.value).shape))
        # print("Vector of 1s has a shape: {}".format(np.ones((n, 1)).shape))
        first_term  = np.dot(self.input_data, self.weights.value)
        second_term = np.dot(np.ones((n, 1)),self.bias.value)
        output = np.dot(self.input_data, self.weights.value) + np.dot(np.ones((n, 1)),self.bias.value)
        # print("Weights are:{}".format(self.weights.value))
        # print("Output(U):{}".format(output))
        self.output = output
        # print("Shape of the forward output (U): {}".format(output.shape))
        ##########            end           ##########

        return output

    def backward(self, delta_n):
        # print("Inside of backward propagation of FCLayer")
        self.backward_counter+=1
        # print("Calling this function for the {}th time".format(self.backward_counter))

        """
        If self.weights.shape = (in, out), then
        :param delta_n:         (numpy.ndarray, shape=[batch_size, out]) the delta from the next layer
        :return delta:          (numpy.ndarray, shape=[batch_size, in]) delta to be passed to the previous layer
        """
        # print("Input delta_n shape: {}".format(delta_n.shape))
        ########## (E2) Your code goes here ##########
        delta = np.dot(delta_n, self.weights.value.T)
        # print("Shape of self.output is: {}".format(self.output.shape))
        dEdW = np.dot(self.output.T, delta).T #H^{l-1}*\delta^l
        # print("Shape of 1s vector:{}".format(np.ones((list(delta_n.shape)[0],1)).shape))
        dEdb = np.dot(np.ones((list(delta_n.shape)[0],1)).T, delta_n)
        #dEdb^L = 1^T\delta^L where delta^L is of dimensions n x m^L
        ##########            end           ##########

        # print("Output delta shape:{}".format(delta.shape))
        # print("Calculated dEdW shape:{}".format(dEdW.shape))
        # print("Calculated dEdb shape:{}".format(dEdb.shape))

        # Store gradients
        self.weights.grad = dEdW
        self.bias.grad = dEdb
        return delta
