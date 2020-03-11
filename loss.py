import numpy as np
from utils import softmax
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def loss(self, pred, target):
        """
        Computes the loss function values by comparing pred and target
        :param pred:        (numpy.ndarray) the output of the output layer (after activation in MSELoss, whereas
                                            before activation in CrossEntropyLoss)
        :param target:      (numpy.ndarray) the labels
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def diff_loss(self, pred, target):
        """
        Computes the derivative of the loss function, i.e., delta
        """
        raise NotImplementedError

class MSELoss(Loss):
    def loss(self, pred, target):
        ########## (E4) Your code goes here ##########
        loss =np.square(np.subtract(target,pred)).mean()
        return loss
        ##########            end           ##########

    def diff_loss(self, pred, target):
        ########## (E4) Your code goes here ##########
        delta = (-2//len(target))* (target-pred)
        return delta
        ##########            end           ##########


class CrossEntropyLoss(Loss):
    """
    This class combines the cross entropy loss with the softmax outputs as done in the PyTorch implementation.
    The return value of self.diff_loss will then be directly handed over to the output FCLayer
    """
    def loss(self, pred, target):
        pred = softmax(pred)
        ########## (E4) Your code goes here ##########
        loss = -np.sum([target[i]*np.log(pred[i]) for i in range(len(target))])
        return loss
        ##########            end           ##########

    def diff_loss(self, pred, target):
        ########## (E4) Your code goes here ##########
        pred = softmax(pred)
        delta = target - pred
        return delta
        ##########            end           ##########

