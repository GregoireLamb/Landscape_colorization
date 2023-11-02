import torch

from src.util import *
from unittest import TestCase


class Test(TestCase):
    def test_ab2class(self):
        tensor = torch.tensor([[[[0.11, 0.21],
                                 [0.01, 0.61]],
                                [[0.31, 0.41],
                                 [0.41, 0.61]]]]) # torch.Size([1, 2, 2, 2])
        classes = ab2class(tensor)
        print(classes)

        assert(classes[0, 0, 0] == 13)
        assert(classes[0, 0, 1] == 24)
        assert(classes[0, 1, 0] == 4)
        assert(classes[0, 1, 1] == 66)

        return 0

    def test_class2ab(self):
        classes = torch.tensor([[[0, 25],
                                 [34, 99]]])  # torch.Size([1, 2, 2])
        tensor = class2ab(classes)

        #A
        assert(tensor[0, 0, 0, 0] == 0.5)
        assert(tensor[0, 0, 0, 1] == 2.5)
        assert(tensor[0, 0, 1, 0] == 3.5)
        assert(tensor[0, 0, 1, 1] == 9.5)
        #B
        assert(tensor[0, 1, 0, 0] == 0.5)
        assert(tensor[0, 1, 0, 1] == 5.5)
        assert(tensor[0, 1, 1, 0] == 4.5)
        assert(tensor[0, 1, 1, 1] == 9.5)

        return 0

    def test_prob2class(self):
        tensor = torch.tensor([[[[0.9, 0.1],
                                 [0.1, 0.1]],
                                [[0.1, 0.8],
                                 [0.8, 0.1]],
                                [[0.1, 0.1],
                                 [0.1, 0.8]],
                                [[0.1, 0.1],
                                 [0.1, 0.8]]]]) #torch.Size([1, 4, 2, 2])
        classes = prob2class(tensor)
        assert(len(classes.shape)==3)
        assert(classes[0, 0, 0] == 0)
        assert(classes[0, 0, 1] == 1)
        assert(classes[0, 1, 0] == 1)
        assert(classes[0, 1, 1] == 2)
        return 0

    def test_ab2prob(self):
        tensor = torch.tensor([[[[0.11, 0.21],
                                 [0.01, 0.61]],
                                [[0.31, 0.41],
                                 [0.41, 0.61]]]])
        prob = ab2prob(tensor)
        assert(len(prob.shape) == 4)
        assert(prob[0, 66, 1, 1] == 1)
        assert(prob[0, 55, 1, 1] == 0)
        assert(prob[0, 24, 0, 1] == 1)
        return 0
