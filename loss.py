import numpy as np
from utils import softmax
from sklearn.metrics import log_loss
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
        loss =np.sum((pred - target)**2) / target.size
        # print("loss from MSE loss:{}".format(loss))
        # print()
        return loss
        ##########            end           ##########

    def diff_loss(self, pred, target):
        ########## (E4) Your code goes here ##########
        # delta = (-2//len(target))* (target-pred)
        delta = target - pred
        # print("delta from derivative of MSE Loss: {}".format(delta))
        return delta
        ##########            end           ##########


class CrossEntropyLoss(Loss):
    # print("Cross entropy loss instance created")
    """
    This class combines the cross entropy loss with the softmax outputs as done in the PyTorch implementation.
    The return value of self.diff_loss will then be directly handed over to the output FCLayer
    """

    def loss(self, pred, target):
        eps = 1e-15
        # print("Inside loss method from CrossEntropyLoss class .........")
        # print("Shape of pred:{}\n Shape of target:{}".format(pred.shape, target.shape))
        # print("Calculating cross entropy loss from pred:{} and target:{}".format(pred, target))
        # pred = softmax(pred)
        # pred = np.clip(pred, eps, 1 - eps)
        ########## (E4) Your code goes here ##########

        pred = np.clip(pred, eps, 1. - eps)
        n = pred.shape[0]
        ce = -np.sum(target * np.log(pred + 1e-9)) / n

        return ce

        # def logloss(y_true, y_pred, eps=1e-15):
        #     pred = np.clip(pred, eps, 1 - eps)
        #     return -(y_true * np.log(y_pred)).sum(axis=1).mean()
        ##########            end           ##########

    #Source: https://deepnotes.io/softmax-crossentropy#cross-entropy-loss
    #Source: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy

    def diff_loss(self, pred, target):
        # print("Inside diff_loss method from CrossEntropyLoss class")
        # print("Shape of pred:{}\n Shape of target:{}".format(pred.shape, target.shape))
        # print("Calculating diff_loss for Cross entropy from pred:{} and target:{}".format(pred, target))
        ########## (E4) Your code goes here ##########
        # m = target.shape[0]
        # grad = softmax(pred)
        # grad[range(m), target] -= 1
        # grad = grad / m

        # pred = softmax(pred)
        delta = target - pred
        # print("Calculated diff_loss delta:{}".format(delta))
        # print("Shape of output from dif_loss method in Loss class: {}".format(delta.shape))
        return delta
        ##########            end           ##########

