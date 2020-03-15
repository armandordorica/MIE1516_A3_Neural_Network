import numpy as np
from utils import softmax
from sklearn.metrics import log_loss
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.autograd import Variable

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
        delta = pred-target
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
        #must return a scalar

        #option 1 from Grosse's notes
        #pred is the matrix of logits and np.logaddexp takes care of converting it to softmax
        loss1 = -np.sum(np.dot(target.T, np.logaddexp(0, -pred)) + np.dot((1-target).T, np.logaddexp(0, pred)))

        #option 2 from https://deepnotes.io/softmax-crossentropy#cross-entropy-loss
        # X = pred
        # y = target
        # y = y.argmax(axis=1)
        # m = y.shape[0]
        # p = softmax(X)
        # print("m is:{}".format(m))
        # We use multidimensional array indexing to extract
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        # log_likelihood = -np.log(p[range(m), y])
        # loss2 = np.sum(log_likelihood) / m

        #option 3
        # pred = softmax(pred)
        # loss3 = np.sum(-np.dot(target.T, np.log(pred)))/m

        # loss3 = np.sum(-target * np.log(pred))
        # print("option 3 loss:{}\n".format(loss3/m))


        loss_torch = nn.CrossEntropyLoss()
        # print("target[0]:{}\n".format([target[0]]))
        # print("pred[0]:{}\n".format([pred[0]]))
        target = Variable(torch.LongTensor([target[0]]), requires_grad=False)

        pred = Variable(torch.Tensor([pred[0]]))

        loss_torch = loss_torch(pred, torch.max(target, 1)[1])
        #https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216
        # print("pytorch loss:{}".format(loss_torch))
        return loss_torch


    #Source: https://deepnotes.io/softmax-crossentropy#cross-entropy-loss
    #Source: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy

    def diff_loss(self, pred, target):
        # print("Inside diff_loss method from CrossEntropyLoss class")
        # print("Shape of pred:{}\n Shape of target:{}".format(pred.shape, target.shape))
        # print("Calculating diff_loss for Cross entropy from pred:\n{}..................\n target\n:{}".format(pred, target))
        ########## (E4) Your code goes here ##########
        # m = target.shape[0]
        # grad = softmax(pred)
        # grad[range(m), target] -= 1
        # grad = grad / m

        pred = softmax(pred)
        delta = pred-target
        # print("Shape of output from dif_loss method in Loss class: {}".format(delta.shape))
        return delta
        ##########            end           ##########

