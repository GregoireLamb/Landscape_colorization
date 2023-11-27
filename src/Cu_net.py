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
  def __init__(self, n_class=128):
    super(Cu_net, self).__init__()
    self.name = "Cu_net"
    self.n_class = 105

    # output_size = (input_size - kernel_size + 2*padding) / stride + 1
    conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)#256
    conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)#256
    conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)#128
    conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)#128
    conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)#64
    conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)#64
    conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False)#32
    conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)#32

    # output_size = strides * (input_size-1) + kernel_size - 2*padding
    deconv1_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)#32
    deconv1_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)#64
    deconv2_1 = nn.ConvTranspose2d(in_channels=256+256, out_channels=self.n_class, kernel_size=4, stride=2, padding=1, bias=False)#128
    deconv2_2 = nn.ConvTranspose2d(in_channels=self.n_class, out_channels=self.n_class, kernel_size=3, stride=1, padding=1, bias=False)#128
    deconv3_1= nn.ConvTranspose2d(in_channels=self.n_class+128, out_channels=self.n_class, kernel_size=3, stride=1, padding=1, bias=False)#128
    deconv3_2= nn.ConvTranspose2d(in_channels=self.n_class, out_channels=self.n_class, kernel_size=3, stride=1, padding=1, bias=False)#128
    deconv4_1= nn.ConvTranspose2d(in_channels=self.n_class+64, out_channels=self.n_class, kernel_size=3, stride=1, padding=1, bias=False)#128
    deconv4_2= nn.ConvTranspose2d(in_channels=self.n_class, out_channels=self.n_class, kernel_size=3, stride=1, padding=1, bias=False)#128

    self.conv1 = nn.Sequential(conv1_1, nn.BatchNorm2d(64), nn.Sigmoid(),
                               conv1_2, nn.BatchNorm2d(64), nn.Sigmoid())
    self.conv2 = nn.Sequential(conv2_1, nn.BatchNorm2d(128), nn.Sigmoid(),
                               conv2_2, nn.BatchNorm2d(128), nn.Sigmoid())
    self.conv3 = nn.Sequential(conv3_1, nn.BatchNorm2d(256), nn.Sigmoid(),
                               conv3_2, nn.BatchNorm2d(256), nn.Sigmoid())

    self.deconv1 = nn.Sequential(deconv1_1, nn.BatchNorm2d(512), nn.Sigmoid(),
                                 deconv1_2, nn.BatchNorm2d(256), nn.Sigmoid())
    self.deconv2 = nn.Sequential(deconv2_1, nn.BatchNorm2d(self.n_class), nn.Sigmoid(),
                                 deconv2_2, nn.BatchNorm2d(self.n_class), nn.Sigmoid())
    self.deconv3 = nn.Sequential(deconv3_1, nn.BatchNorm2d(self.n_class), nn.Sigmoid(),
                                 deconv3_2, nn.BatchNorm2d(self.n_class), nn.Sigmoid())
    self.deconv4 = nn.Sequential(deconv4_1, nn.BatchNorm2d(self.n_class), nn.Sigmoid(),
                                 deconv4_2, nn.BatchNorm2d(self.n_class), nn.Sigmoid())


  def forward(self, input):
      conv1 = self.conv1(input)
      conv2 = self.conv2(conv1)
      conv3 = self.conv3(conv2)
      conv4 = self.conv4(conv3)

      deconv1 = self.deconv1(conv4)
      deconv2 = self.deconv2(torch.cat([deconv1, conv3], dim=1))
      deconv3 = self.deconv3(torch.cat([deconv2, conv2], dim=1))
      deconv4 = self.deconv4(torch.cat([deconv3, conv1], dim=1))

      return deconv4



