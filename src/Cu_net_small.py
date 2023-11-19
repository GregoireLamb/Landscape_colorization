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


class Cu_net_small(nn.Module):
  def __init__(self, n_class=128):
    super(Cu_net_small, self).__init__()
    self.name = "Cu_net_small"
    self.n_class = 105

    # output_size = (input_size - kernel_size + 2*padding) / stride + 1
    conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)#256
    conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)#128
    conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)#64

    # output_size = strides * (input_size-1) + kernel_size - 2*padding
    deconv1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)#64
    deconv2_1 = nn.ConvTranspose2d(in_channels=128, out_channels=self.n_class, kernel_size=3, stride=1, padding=1, bias=False)#64
    deconv_3= nn.ConvTranspose2d(in_channels=self.n_class, out_channels=self.n_class, kernel_size=3, stride=1, padding=1, bias=False)#256

    # upx1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    upx2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    self.net = nn.Sequential(
        conv1_1, nn.BatchNorm2d(64), nn.Sigmoid(),
        conv2_1, nn.BatchNorm2d(128), nn.Sigmoid(),
        conv3_1, nn.BatchNorm2d(256), nn.Sigmoid(),
        deconv1_1, nn.BatchNorm2d(128), nn.Sigmoid(),
        # upx1, nn.BatchNorm2d(128), nn.Sigmoid(),
        deconv2_1, nn.BatchNorm2d(self.n_class), nn.Sigmoid(),
        upx2, nn.BatchNorm2d(self.n_class), nn.Sigmoid(),
        deconv_3, nn.BatchNorm2d(self.n_class), nn.Sigmoid(),
    )

  def forward(self, input):
    return self.net(input)



