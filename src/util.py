import math
from typing import Dict, Union, Any

import scipy.stats
import torch
from logging import warn, warning

class2mapping = torch.tensor([
    25, 26, 27, 40, 41, 42, 51, 52, 53, 54,55, 56, 57, 64, 65, 66, 67, 68, 69, 70,71, 72, 79, 80, 81, 82, 83, 84,
    85, 86,87, 93, 94, 95, 96, 97, 98, 99, 100, 101,102, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 123, 124, 125, 126, 127, 128, 129, 130, 131,132, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 146, 147, 150, 151, 152, 153, 154, 155, 156,157, 158, 159, 160, 165, 166, 167, 168, 169, 170,
    171, 172, 173, 174, 175, 180, 182, 183, 184, 185,186, 187, 188, 189, 190
], dtype=torch.long).cuda()

mapping2class = torch.full((225,), 104, dtype=torch.long).long().cuda()
mapping2class[class2mapping] = torch.arange(len(class2mapping)).long().cuda()
# print(mapping2class)

def class2ab(te, n_classes=100):
    """
    :param te: torch.Size([1, 256, 256]): batch_size, height, width (value = classes)
    :return: torch.Size([1, 2, 256, 256]): batch_size, ab, height, width
    """
    ab = torch.zeros(te.shape[0], 2, te.shape[1], te.shape[2], dtype=torch.float32)

    if n_classes == 105:
        where = torch.where(te == 106)  # 106 is the error class make a and b  0
        te[where] = 104
        te = class2mapping[te]
        a_channel, b_channel = ((te // 15) / 15 + 1/30, (te % 15) / 15 + 1/30)
    else :
        a_channel, b_channel = ((te // 10) / 10 + 0.05, (te % 10) / 10 + 0.05)

    # Assign values to the corresponding channels in the output tensor
    ab[:, 0, :, :] = a_channel
    ab[:, 1, :, :] = b_channel
    return ab


def ab2prob(te, n_classes=100, neighbooring_class=5):
    """
    :param te: torch.Size([1, 2, 256, 256]): batch_size, ab, height, width
    :return: torch.Size([1, 100, 256, 256]): batch_size, Q_truth, height, width
    1/0 for now
    """
    batch_size, _, height, width = te.shape
    #
    # a_values = te[:, 0, :, :].unsqueeze(1) # batch, height, width
    # b_values = te[:, 1, :, :].unsqueeze(1)

    # main_classes = a_b_float2class(a_values, b_values)# batch, height, width
    # neighbor_offsets = torch.tensor([0]).unsqueeze(1).unsqueeze(1).cuda()  # torch.Size([5, 1, 1])
    # neighbor_offsets = torch.tensor([-11, -10, -9, -1, 0, 1, 9, 10, 11]).unsqueeze(1).unsqueeze(1).cuda()  # torch.Size([5, 1, 1])
    # neighbor_offsets = torch.tensor([0, 1, 10]).unsqueeze(1).unsqueeze(1).cuda()  # torch.Size([2, 1, 1]) TODO adapt for extrem values
    # neighbors_classes = main_classes.float() + neighbor_offsets
    # neighbors = torch.clamp(neighbors_classes, min=0, max=n_classes - 1).long()  # Convert to integer tensor

    # prob = torch.zeros(batch_size, n_classes, height, width).cuda()
    # print("a_b_float2class(a_values, b_values).long().shape()", torch.Tensor(a_b_float2class(a_values, b_values).cpu()).shape())

    # MAIN CLASS
    prob = torch.zeros(batch_size, n_classes, height, width).cuda()
    prob[torch.arange(batch_size).unsqueeze(1), ab2class(te, n_classes=n_classes), torch.arange(height).unsqueeze(0).unsqueeze(2), torch.arange(width).unsqueeze(0).unsqueeze(1)] = 1
    # TODO WARNIG HERE GIB CHANGE FROM BELLOW TO ABOVE
    # prob[torch.arange(batch_size).unsqueeze(1), ab2class(a_values, b_values), torch.arange(height).unsqueeze(0).unsqueeze(2), torch.arange(width).unsqueeze(0).unsqueeze(1)] = 1

    # # Assign values from the neighbors tensor to the corresponding positions in the probability tensor
    # prob.scatter_add_(1, neighbors,
    #                   gaussian(a_values, b_values, neighbors_classes // 10 / 10 + 0.05, neighbors_classes % 10 / 10 + 0.05))
    return prob

def ab2class(ab, n_classes=100):
    """
    :param ab: values in [0,1]
    :param n_classes: default 313
    :return: class [0,n_classes-1]
    """
    if n_classes == 105:
        mapping = (torch.round(ab[:, 0, :, :]*14)*15 + torch.round(ab[:, 1, :, :]*14)).clone().long() # class in a 15*15 grid
        return mapping2class[mapping]
    return (torch.floor(ab[:, 0, :, :] * 10) * 10 + torch.floor(ab[:, 1, :, :]*10)).clone().long()

def prob2ab(te, n_classes=100, neighbooring_class=4, temperature=1, prob_max=True):
    """
    :param tensor: te: torch.Size([1, 100, 256, 256]): batch_size, Q, height, width (value = prob, [0,1]
    :return: ab: torch.Size([1, 2, 256, 256]): batch_size, ab, height, width (value = classes)
    """
    if prob_max:
        return class2ab(torch.argmax(te, dim=1), n_classes=n_classes)
    # batch_size, _, height, width = te.shape
    # # Sum along the Q dimension and divide by the sum of the input tensor along the Q dimension
    # ab = torch.zeros(batch_size, 2, height, width, dtype=torch.float32).cuda()
    # if n_classes == 105:
    #     ab = class2ab(torch.argmax(te, dim=1), n_classes=n_classes)
    #     # ab[:, 0, :, :] = torch.floor(torch.argmax(te, dim=1) / 15) / 15 + 1/30
    #     # ab[:, 1, :, :] = torch.argmax(te, dim=1) % 15 / 15 + 1/30
    # else:
    #     ab[:, 0, :, :] = torch.argmax(te, dim=1) // 10 / 10 + 0.05
    #     ab[:, 1, :, :] = torch.argmax(te, dim=1) % 10 / 10 + 0.05
    # return ab

# def prob2ab(te, n_classes=100, neighbooring_class=4, temperature=0.38):
#     """
#     :param tensor: te: torch.Size([1, 100, 256, 256]): batch_size, Q, height, width (value = prob, [0,1]
#     :return: ab: torch.Size([1, 2, 256, 256]): batch_size, ab, height, width (value = classes)
#     """
#     batch_size, _, height, width = te.shape
#
#     multiplication_factors_a = torch.tensor([(i) * 10 + 5 for i in range(10) for j in range(10)],
#                                             dtype=torch.float32).cuda()
#     multiplication_factors_b = torch.tensor([(i) * 10 + 5 for j in range(10) for i in range(10)],
#                                             dtype=torch.float32).cuda()
#
#     # Expand multiplication factors to match tensor shape
#     multiplication_factors_a = multiplication_factors_a.view(1, n_classes, 1, 1)
#     multiplication_factors_b = multiplication_factors_b.view(1, n_classes, 1, 1)
#
#     # "un gaussian" distances
#     # te = un_gaussian(te) #torch.Size([1, 100, 256, 256])
#
#     # Multiply tensor by factors
#     a_mult = te * multiplication_factors_a
#     b_mult = te * multiplication_factors_b
#
#     # Sum along the Q dimension and divide by the sum of the input tensor along the Q dimension
#     ab = torch.zeros(batch_size, 2, height, width, dtype=torch.float32).cuda()
#     ab[:, 0, :, :] = torch.exp(torch.log(torch.sum(a_mult, dim=1))/torch.tensor(temperature)) / torch.exp(torch.log((torch.sum(te, dim=1) * n_classes))/torch.tensor(temperature))
#     ab[:, 1, :, :] = torch.sum(b_mult, dim=1) / (torch.sum(te, dim=1) * n_classes)
#
#     return ab

def gaussian(a,b, x, y, sig = 0.1):
    """
    :param a:  a truth
    :param b: b truth
    :param x: a_class
    :param y: b_class
    :param sig: gaussian sigma
    :return: distance between a,b and x,y
    """
    dist = torch.sqrt((a - x) ** 2 + (b - y) ** 2) # euclidean distance
    coefficient = 1 / (math.sqrt(2 * math.pi) * sig)
    exponent = -(dist ** 2) / (2 * sig ** 2)
    return coefficient * torch.exp(exponent)/2

def un_gaussian(dist, sig=0.1):
    """
    :param dist: distance between a,b and x,y (gaussian)
    :param sig: gaussian sigma
    :return: euclidean distance
    """
    coefficient = 1 / (math.sqrt(2 * math.pi) * sig)
    non_null_mask = (dist != 0)  # Create a mask for non-null values
    non_null_indices = non_null_mask.nonzero(as_tuple=False).squeeze(1)#.unsqueeze(1) # Get indices of non-null values
    non_null_dist = dist[non_null_mask]  # Apply mask to obtain non-null distances

    edist = torch.sqrt(-2 * sig ** 2 * torch.log(non_null_dist / coefficient) / torch.log(torch.tensor(2.7182818284)))
    euclidean_distance = torch.zeros_like(dist)

    euclidean_distance[non_null_indices[:, 0],
                        non_null_indices[:, 1],
                        non_null_indices[:, 2],
                        non_null_indices[:, 3]] = edist

    return euclidean_distance

