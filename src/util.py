from PIL import Image
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torchvision import transforms
import glob
import tensorflow as tf

from skimage import color


def load_img(path):
    # Load an image from disk and return it as a numpy array
    img = np.asarray(Image.open(path))
    return img

def turn_to_grayscale(img):
    # Turn an RGB image into a grayscale image
    wR = 0.290
    wG = 0.587
    wB = 0.114

    grayscale_intensity = wR * img[:, :, 0] + wG * img[:, :, 1] + wB * img[:, :, 2]
    return grayscale_intensity

def one_batch_data(path):
    data_loader = torch.utils.data.DataLoader(path,
                                              batch_size=1,
                                              shuffle=True)
    return data_loader
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


def load2(path):
    X_train = np.load(path)
    data = X_train.astype(np.float64)
    data = 255 * data
    X_train = data.astype(np.uint8)
    random_image = random.randint(0, len(X_train))
    plt.imshow(X_train[random_image])
    plt.title(f"Training example #{random_image}")
    plt.axis('off')
    plt.show()


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

if __name__ == '__main__':
    load2('imgds.npy')

    # img = load_img('../data_square/00000000_(2).jpg')
    # plt.imshow(img)
    # plt.show()
    #
    # grayscale = turn_to_grayscale(img)
    # plt.imshow(grayscale, cmap='gray')
    # plt.show()
