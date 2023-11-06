import math
import torch
from logging import warn, warning

def class2ab(te):
    """
    :param te: torch.Size([1, 256, 256]): batch_size, height, width (value = classes)
    :return: torch.Size([1, 2, 256, 256]): batch_size, ab, height, width
    """
    ab = torch.zeros(te.shape[0], 2, te.shape[1], te.shape[2], dtype=torch.float32)
    for s in range(te.shape[0]):
        for i in range(te.shape[1]):
            for j in range(te.shape[2]):
                #TODO adapt for variable amount of classes
                ab[s, 0, i, j], ab[s, 1, i, j] = class2a_b_float(te[s, i, j].item())
    return ab

def prob2class(te):
    """
    :param tensor: te: torch.Size([1, 100, 256, 256]): batch_size, Q, height, width (value = prob, [0,1]
    :return: class: torch.Size([1, 256, 256]): batch_size, height, width (value = classes)
    equal prob -> smallest class
    """
    classes = torch.zeros(te.shape[0], te.shape[2], te.shape[3], dtype=torch.float32)

    for s in range(te.shape[0]):
        for i in range(te.shape[2]):
            for j in range(te.shape[3]):
                #TODO adapt for variable amount of classes
                classes[s, i, j] = torch.argmax(te[s, :, i, j], dim=0)
    return classes

def ab2class(te):
    """
    :param te: torch.Size([1, 2, 256, 256]): batch_size, ab, height, width
    :return: torch.Size([1, 256, 256]): batch_size, height, width (value = classes)
    """
    #TODO adapt shape
    prediction_grid_size = 10
    classes = torch.zeros(te.shape[0],te.shape[2], te.shape[3], dtype=torch.long)
    for s in range(te.shape[0]):
        for i in range(te.shape[2]):
            for j in range(te.shape[3]):
                classes[s, i, j] =a_b_float2class(te[s, 0, i, j].item(), te[s, 1, i, j].item())
    return classes

def ab2prob(te, n_classes=100):
    """
    :param te: torch.Size([1, 2, 256, 256]): batch_size, ab, height, width
    :return: torch.Size([1, 100, 256, 256]): batch_size, Q_truth, height, width
    1/0 for now
    """
    prob = torch.zeros(te.shape[0], n_classes, te.shape[2], te.shape[3], requires_grad=True)
    for s in range(te.shape[0]):
        for i in range(te.shape[2]):
            for j in range(te.shape[3]):
                prob[s, a_b_float2class(te[s, 0, i, j].item(), te[s, 1, i, j].item()), i, j] = 1
    return prob

def a_b_float2class(a,b, n_classes=313):
    """
    :param a: [0,1]
    :param b: [0,1]
    :param n_classes: default 313
    :return: class [0,n_classes-1]
    """
    #TODO adapt shape
    if not (0 <= a <= 1 and 0 <= b <= 1) :
        warning("a or b not in [0,1], class = -1")
        return -1

    return math.floor(a * 10) * 10 + math.floor(b*10)

def class2a_b_float(cl, n_classes=313):
    """
    :param cl: [0,n_classes-1]
    :param n_classes: default 313
    :return: (a,b) [0,1]
    """
    #TODO adapt shape
    # if n_classes == 200:
    #     return ((cl // 20 + 0.25) / 10, (cl % 10 + 0.5) / 20)
    return ((cl // 10) / 10, (cl % 10) / 10 )



