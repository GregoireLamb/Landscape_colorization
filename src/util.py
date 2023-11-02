import math

from PIL import Image
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms
import glob
import tensorflow as tf

from skimage import color


# def val2class_baseline(tensor):
#     assert(len(tensor.shape) == 4)
#     assert(tensor.shape[1] == 2)
#
#     prediction_grid_size = 10
#     truth = torch.zeros(tensor.shape[0],1 , tensor.shape[2], tensor.shape[3], dtype=torch.long)
#
#     for s in range(tensor.shape[0]):
#         for i in range(tensor.shape[2]):
#             for j in range(tensor.shape[3]):
#                 truth[s, 0, i, j] = 1+int(tensor[s, 0, i, j].item() * 90)+int(tensor[s, 1, i, j].item() * 9)
#     return torch.flatten(truth, start_dim=1)


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
                ab[s, 0, i, j] = te[s, i, j].item()//10+0.5
                ab[s, 1, i, j] = te[s, i, j].item()%10+0.5
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



#
# def load_img(path):
#     # Load an image from disk and return it as a numpy array
#     img = np.asarray(Image.open(path))
#     return img

# def turn_to_grayscale(img):
#     # Turn an RGB image into a grayscale image
#     wR = 0.290
#     wG = 0.587
#     wB = 0.114
#
#     grayscale_intensity = wR * img[:, :, 0] + wG * img[:, :, 1] + wB * img[:, :, 2]
#     return grayscale_intensity

# def one_batch_data(path):
#     data_loader = torch.utils.data.DataLoader(path,
#                                               batch_size=1,
#                                               shuffle=True)
#     return data_loader
    # paths = glob.glob(path)
    # print(paths)
    # img = []
    # grayscale = []
    # i = 0
    # for path in paths:
    #     # add the image to the batch in a list
    #     image = load_img(path)
    #     img.append(transforms.ToTensor()([None, image]))
    #     grayscale.append(transforms.ToTensor()(turn_to_grayscale([None, image])))

    # return grayscale, img


# def load2(path):
#     X_train = np.load(path)
#     data = X_train.astype(np.float64)
#     data = 255 * data
#     X_train = data.astype(np.uint8)
#     random_image = random.randint(0, len(X_train))
#     plt.imshow(X_train[random_image])
#     plt.title(f"Training example #{random_image}")
#     plt.axis('off')
#     plt.show()

