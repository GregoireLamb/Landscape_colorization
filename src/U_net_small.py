import torch
import torch.nn.functional as F
import torch.nn as nn

class U_net_small(nn.Module):
    def __init__(self, n_class=105):
        super(U_net_small, self).__init__()
        self.name = "U_net_small"
        self.n_class = n_class
        self.custom_kernel = nn.Parameter(torch.Tensor([[[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]]))

        # output_size = (input_size - kernel_size + 2*padding) / stride + 1
        conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        conv1_1.weight = self.custom_kernel
        conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        conv2_1.weight = self.custom_kernel
        conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False)
        conv3_1.weight = self.custom_kernel

        deconv1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        deconv2_1 = nn.ConvTranspose2d(in_channels=128, out_channels=self.n_class, kernel_size=3, stride=1, padding=1, bias=False)
        deconv_3 = nn.ConvTranspose2d(in_channels=self.n_class, out_channels=self.n_class, kernel_size=3, stride=1, padding=1, bias=False)

        upx2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.net = nn.Sequential(
            conv1_1, nn.BatchNorm2d(64), nn.Sigmoid(),
            conv2_1, nn.BatchNorm2d(128), nn.Sigmoid(),
            conv3_1, nn.BatchNorm2d(256), nn.Sigmoid(),
            deconv1_1, nn.BatchNorm2d(128), nn.Sigmoid(),
            deconv2_1, nn.BatchNorm2d(self.n_class), nn.Sigmoid(),
            upx2, nn.BatchNorm2d(self.n_class), nn.Sigmoid(),
            deconv_3, nn.BatchNorm2d(self.n_class), nn.Sigmoid(),
        )

    def forward(self, input):
        return self.net(input)

