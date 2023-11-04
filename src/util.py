import math
import torch


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
                ab[s, 0, i, j] = (te[s, i, j].item()//10+0.5)/10
                ab[s, 1, i, j] = (te[s, i, j].item()%10+0.5)/10
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
                classes[s, i, j] = math.floor(te[s, 0, i, j].item() * 10) * 10 + math.floor(te[s, 1, i, j].item() * 10)
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
                prob[s, math.floor(te[s, 0, i, j].item() * 10) * 10 + math.floor(te[s, 1, i, j].item() * 10) , i, j] = 1
    return prob

def a_b_float2class(a,b, n_classes=313):
    """
    :param a: [0,1]
    :param b: [0,1]
    :param n_classes: default 313
    :return: class [0,n_classes-1]
    """
    #TODO adapt shape
    return math.floor(value * 10) * 10


class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial'''
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


