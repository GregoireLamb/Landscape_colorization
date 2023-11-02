# For plotting
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
# For our model
import torchvision.models as models
from torchvision import datasets, transforms
# For utilities
import os, shutil, time


class Cu_net(nn.Module):
  def __init__(self, input_size=128):
    super(Cu_net, self).__init__()
    # self.custom_kernel = torch.tensor(64, 1, 3,3,[[1, 0, 1],
    #                                    [0, 1, 0],
    #                                    [1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Kernel from paper2

    conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, bias=False)
    conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, bias=False)
    conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, bias=False)
    conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, bias=False)
    # conv6_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1)
    deconv_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, bias=False)
    deconv_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, bias=False)
    deconv_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, bias=False)
    deconv_4 = nn.ConvTranspose2d(in_channels=64, out_channels=100, kernel_size=3, stride=1, bias=False)

    self.net = nn.Sequential(
        conv1, nn.BatchNorm2d(64), nn.Sigmoid(),
        # conv2, nn.BatchNorm2d(128), nn.Sigmoid(),
        # conv3, nn.BatchNorm2d(256), nn.Sigmoid(),
        # # conv4, nn.BatchNorm2d(512), nn.Sigmoid(),
        # # deconv_1, nn.BatchNorm2d(256), nn.Sigmoid(),
        # deconv_2, nn.BatchNorm2d(128), nn.Sigmoid(),
        # deconv_3, nn.BatchNorm2d(64), nn.Sigmoid(),
        deconv_4, nn.BatchNorm2d(100), nn.Sigmoid()
    )


  def forward(self, input):
    return self.net(input)
