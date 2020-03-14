import numpy as np
import numpy.linalg as LA
from abc import ABC, abstractmethod
import math

class Optimizer(ABC):
    def __init__(self, parameters, lr, clipvalue):
        """
        A base class for optimizers. An Optimizer object should be initialized with trainable Parameter objects
        of the model
        :param parameters:          (list) a list of param.Parameter objects of the neural net
        :param lr:                  (float) learning rate
        :param clipvalue:           (float or None) threshold value when clipping gradients (optional)
        """
        self.param_lst = parameters
        self.learning_rate = lr
        self.clipvalue = clipvalue

    @abstractmethod
    def step(self):
        """
        A method which actually updates all the values of Parameters.
        Each Optimizer object should have its own implementation of this method.
        """
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, parameters, lr,
                 clipvalue=None,
                 momentum=False,
                 nesterov=False,
                 mu=0.9,
                 **kwargs):
        """
        A SGD optimizer (optionally with nesterov/momentum).
        :param parameters:          (list) list of Parameter
        :param lr:                  (float) learning rate
        :param clipvalue:           (float or None) threshold value when clipping gradients
        :param momentum:            (bool) whether to use momentum
        :param nesterov:            (bool) whether to use Nesterov Accelerated Gradient
        :param mu:                  (float) momentum parameter used for 'momentum' or 'nesterov' (default: 0.9)
        :param kwargs:              (additional paramters)
        """
        # Link the Optimizer with Parameters, and set up some attributes
        super(SGD, self).__init__(parameters, lr, clipvalue=clipvalue)
        if nesterov and momentum:
            raise ValueError("Either momentum or nesterov should be turned on, not both.")

        if nesterov or momentum:
            # the momentum parameter *mu*
            self.mu = mu

            # initialize velocity of parameters to zero
            for param in self.param_lst:
                param.velocity = np.zeros(param.shape, dtype=np.float32)

        self.nesterov = nesterov
        self.momentum = momentum
        self.kwargs = kwargs

        # which norm to use in gradient clipping (default is 2-norm)
        self.ord = self.kwargs['ord'] if 'ord' in self.kwargs else 2

    def step(self):
        # clip gradient values (see torch.nn.utils.clip_grad_norm_ function)
        total_norm = 0
        if self.clipvalue is not None:
            for param in self.param_lst:
                param_norm = LA.norm(param.grad, ord=self.ord)
                total_norm += param_norm ** self.ord
            total_norm = total_norm ** (1. / self.ord)
            clip_coef = self.clipvalue / (total_norm + 1e-6)
            if clip_coef < 1:
                for param in self.param_lst:
                    param.grad *= clip_coef

        # Loop through Parameters and update their values
        for param in self.param_lst:
            # no updates happen when there's no gradient information
            if param.grad is None:
                continue

            # momentum or Nesterov Accelerated Gradient update
            if self.momentum or self.nesterov:
                v_prev = param.velocity.copy()
                if self.momentum:
                    param.velocity = self.mu * v_prev - self.learning_rate * param.grad
                    descent = param.velocity

                # Implement Nesterov accelarted gradient
                ########## (E5) Your code goes here ##########
                else:
                    # print("param.grad is: {}".format(param.grad))
                    # print("v_prev is: {}".format(v_prev))
                    param.velocity =  self.mu * v_prev + param.grad
                    descent = param.grad * self.learning_rate - param.velocity *self.mu * self.learning_rate


                    #reference: https://github.com/hero9968/PaddlePaddle-book/blob/1ff47b284c565d030b198705d5f18b4bd4ce53e5/python/paddle/v2/fluid/tests/test_momentum_op.py
                    # param - grad * learning_rate - \
                    # velocity_out * mu * learning_rate
                    #velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
                    # model[layer] += velocity[layer]
                ##########            end           ##########

            # ordinary gradient descent (w/o momentum)
            else:
                descent = -self.learning_rate * param.grad

            # update
            param.value += descent

class Adam(Optimizer):
    def __init__(self, parameters, lr,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8,
                 clipvalue=None):
        # print("Initializing an Adam optimizer with learning rate: {} ...".format(lr))
        """
        An Adam optimizer
        :param parameters:          (list) list of Paramters of the neural network
        :param lr:                  (float) learning rate
        :param beta1:               (float) Adam specific parameter
        :param beta2:               (float) Adam specific parameter
        :param eps:                 (float) Adam specific parameter
        """
        # Link the Optimizer with Parameters, and set up some attributes
        super(Adam, self).__init__(parameters, lr, clipvalue=clipvalue)


        # Initialize Parameter.m_t and Parameter.v_t: moving average of first and second moments of gradient
        for param in self.param_lst:
            # print("There are {} elements in param_lst".format(len(self.param_lst)))
            param.m_t = np.zeros(param.shape, dtype=np.float32)
            # print("param.m_t: {}".format(param.m_t))
            param.v_t = np.zeros(param.shape, dtype=np.float32)
            # print("param.v_t: {}".format(param.v_t))
            param.step = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self):
        # Loop through Parameters and update their values
        for param in self.param_lst:
            param.step += 1
            bias_correction1 = 1 - self.beta1 ** param.step
            bias_correction2 = 1 - self.beta2 ** param.step

            # print("type of self.beta1 shape is:{}".format(type(self.beta1)))
            # print("self.beta1 = {}".format(self.beta1))
            # print("param.m_t shape is:{}".format(param.m_t.shape))
            # print("param.grad shape is:{}".format(param.grad.shape))
            param.m_t = self.beta1 * param.m_t + (1 - self.beta1) * param.grad
            param.v_t = self.beta2 * param.v_t + (1 - self.beta2) * (param.grad**2)

            denom = np.sqrt(param.v_t) / math.sqrt(bias_correction2) + self.eps
            step_size = self.learning_rate / bias_correction1

            descent = -step_size * param.m_t / denom
            param.value += descent

class RMSProp(Optimizer):
    def __init__(self, parameters, lr,
                 decay_rate=0.9,
                 clipvalue=None,
                 **kwargs):
        """
        A RMSProp optimizer
        :param parameters:          (list) list of Parameter
        :param lr:                  (float) learning rate
        :param decay_rate:          (float) RMSProp specific parameter (default = 0.9)
        :param kwargs: additional arguments
        """
        # Link the Optimizer with Parameters, and set up some attributes
        super(RMSProp, self).__init__(parameters, lr, clipvalue=clipvalue)

        # Initialize Parameter.meansquare which keeps track of the moving average of squared gradients
        for param in self.param_lst:
            param.meansquare = np.zeros(param.shape, dtype=np.float32)

        self.decay_rate = decay_rate

    def step(self):
        # Loop through Parameters and update their values
        for param in self.param_lst:
            param.meansquare = self.decay_rate * param.meansquare + (1 - self.decay_rate) * param.grad**2
            descent = -self.learning_rate * param.grad / (np.sqrt(param.meansquare) + 1e-6)
            param.value += descent

