import numpy as np
from skimage.color import lab2rgb, rgb2lab, rgb2gray
import torch
from torchvision import datasets

class GrayscaleImageFolder(datasets.ImageFolder):
  '''Custom images folder, which converts images to grayscale before loading'''

  def __getitem__(self, index):
      path, target = self.imgs[index]
      img = self.loader(path)
      if self.transform is not None:
          img_original = self.transform(img)
          img_original = np.asarray(img_original)
          img_lab = rgb2lab(img_original)
          img_lab = (img_lab + 128) / 255 #nomarlise to [0,1]
          img_ab = img_lab[:, :, 1:3] # get ab channels
          img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float() # change to pytorch tensor, ab channel - height - width
          # img_original = self.rgbtograyscale(img_original)
          # TODO change L
          img_original = rgb2gray(img_original)
          img_original = torch.from_numpy(img_original).unsqueeze(0).float()
      if self.target_transform is not None:
          target = self.target_transform(target)
      return img_original, img_ab, target

  # def rgbtograyscale(self, rgb):
  #     wR = 0.290
  #     wG = 0.587
  #     wB = 0.114
  #
  #     coeff = np.array([wR, wG, wB])
  #     return rgb @ coeff
