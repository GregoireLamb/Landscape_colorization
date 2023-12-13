import torch
import torch.nn as nn

class Cu_net_small(nn.Module):
  def __init__(self, n_class=128):
    super(Cu_net_small, self).__init__()
    self.name = "Cu_net_small"
    self.n_class = 105
    self.custom_kernel = nn.Parameter(torch.Tensor([[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]))

    # output_size = (input_size - kernel_size + 2*padding) / stride + 1
    conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)#256
    conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)#128
    conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False)#64

    # output_size = strides * (input_size-1) + kernel_size - 2*padding
    deconv1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False)#128
    deconv2_1 = nn.ConvTranspose2d(in_channels=128+128, out_channels=self.n_class, kernel_size=4, stride=2, padding=1, bias=False)#256
    deconv_3= nn.ConvTranspose2d(in_channels=self.n_class+64, out_channels=self.n_class, kernel_size=3, stride=1, padding=1, bias=False)#256

    self.conv1 = nn.Sequential(conv1_1, nn.BatchNorm2d(64), nn.Sigmoid())
    self.conv2 = nn.Sequential(conv2_1, nn.BatchNorm2d(128), nn.Sigmoid())
    self.conv3 = nn.Sequential(conv3_1, nn.BatchNorm2d(256), nn.Sigmoid())

    self.deconv1 = nn.Sequential(deconv1_1, nn.BatchNorm2d(128), nn.Sigmoid())
    self.deconv2 = nn.Sequential(deconv2_1, nn.BatchNorm2d(self.n_class), nn.Sigmoid())
    self.deconv3 = nn.Sequential(deconv_3, nn.BatchNorm2d(self.n_class), nn.Sigmoid())



  def forward(self, input):
      conv1 = self.conv1(input)
      conv2 = self.conv2(conv1)
      conv3 = self.conv3(conv2)

      deconv1 = self.deconv1(conv3)
      deconv2 = self.deconv2(torch.cat([deconv1, conv2], dim=1))
      deconv3 = self.deconv3(torch.cat([deconv2, conv1], dim=1))

      return deconv3
