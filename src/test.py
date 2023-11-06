import os
from unittest import TestCase

import torch

from src.GrayscaleImageFolder import GrayscaleImageFolder
from src.main import to_rgb
from src.util import *

import os
import sys

from alive_progress import alive_bar
from skimage.color import xyz2rgb
from skimage.color.colorconv import _prepare_colorarray, get_xyz_coords

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Cu_net import *
from src.GrayscaleImageFolder import *
from src.util import *
import click


class Test(TestCase):
    def test_ab2class(self):
        tensor = torch.tensor([[[[0.11, 0.21],
                                 [0.01, 0.61]],
                                [[0.31, 0.41],
                                 [0.41, 0.61]]]]) # torch.Size([1, 2, 2, 2])
        classes = ab2class(tensor)
        print(classes)

        assert(classes[0, 0, 0] == 13)
        assert(classes[0, 0, 1] == 24)
        assert(classes[0, 1, 0] == 4)
        assert(classes[0, 1, 1] == 66)

        return 0

    def test_class2ab(self):
        classes = torch.tensor([[[0, 25],
                                 [34, 99]]])  # torch.Size([1, 2, 2])
        tensor = class2ab(classes)

        #A
        assert(tensor[0, 0, 0, 0] == 0.05)
        assert(tensor[0, 0, 0, 1] == 0.25)
        assert(tensor[0, 0, 1, 0] == 0.35)
        assert(tensor[0, 0, 1, 1] == 0.95)
        #B
        assert(tensor[0, 1, 0, 0] == 0.05)
        assert(tensor[0, 1, 0, 1] == 0.55)
        assert(tensor[0, 1, 1, 0] == 0.45)
        assert(tensor[0, 1, 1, 1] == 0.95)

        return 0

    def test_prob2class(self):
        tensor = torch.tensor([[[[0.9, 0.1],
                                 [0.1, 0.1]],
                                [[0.1, 0.8],
                                 [0.8, 0.1]],
                                [[0.1, 0.1],
                                 [0.1, 0.8]],
                                [[0.1, 0.1],
                                 [0.1, 0.8]]]]) #torch.Size([1, 4, 2, 2])
        classes = prob2class(tensor)
        assert(len(classes.shape)==3)
        assert(classes[0, 0, 0] == 0)
        assert(classes[0, 0, 1] == 1)
        assert(classes[0, 1, 0] == 1)
        assert(classes[0, 1, 1] == 2)
        return 0

    def test_ab2prob(self):
        tensor = torch.tensor([[[[0.11, 0.21],
                                 [0.01, 0.61]],
                                [[0.31, 0.41],
                                 [0.41, 0.61]]]])
        prob = ab2prob(tensor)
        assert(len(prob.shape) == 4)
        assert(prob[0, 66, 1, 1] == 1)
        assert(prob[0, 55, 1, 1] == 0)
        assert(prob[0, 24, 0, 1] == 1)
        return 0

    def test_all_colors (self):
        L = 80
        os.makedirs('test_color/gray/', exist_ok=True)
        os.makedirs('test_color/color/', exist_ok=True)
        teL = torch.zeros(1, 256, 256, dtype=torch.float32)
        image = torch.zeros(2, 256, 256, dtype=torch.float32)
        all_ab = torch.zeros(1, 2, 256, 256, dtype=torch.float32)
        for i in range(0, 256):
            for j in range(0, 256):
                all_ab[0, 0, i, j] = i/256
                all_ab[0, 1, i, j] = j/256
                teL[0, i, j] = L/100
        image = class2ab(ab2class(all_ab)).squeeze(0)
        # image = all_ab.squeeze(0)
        for j in range(min(len(image), 5)):
            save_path = {'grayscale': 'test_color/gray/', 'colorized': 'test_color/color/'}
            save_name = 'img2-recolor-L-{}.jpg'.format(L)
            image = image.detach()
            to_rgb(teL.cpu(), ab_input=image.detach().cpu(), save_path=save_path, save_name=save_name)
        return 0

    def test_all_classes (self):
        os.makedirs('test_color/gray/', exist_ok=True)
        os.makedirs('test_color/color/', exist_ok=True)

        teL = torch.zeros(1, 256, 256, dtype=torch.float32)
        classes = torch.zeros(1, 256, 256, dtype=torch.float32)
        image = torch.zeros(2, 256, 256, dtype=torch.float32)
        for i in range(0, 256):
            for j in range(0, 256):
                classes[0, i, j] = int(i/256 *100)
                teL[0, i, j] = j/256

        for i in range(0, 256):
            for j in range(0, 256):
                image[0, i, j], image[1, i, j] = class2a_b_float(classes[0,i,j])

        for j in range(min(len(image), 5)):
            save_path = {'grayscale': 'test_color/gray/', 'colorized': 'test_color/color/'}
            save_name = 'img2-{}-epoch-{}.jpg'.format(43,50)
            image = image.detach()
            to_rgb(teL.cpu(), ab_input=image.detach().cpu(), save_path=save_path, save_name=save_name)
        return 0

    def test_recolorize (self):
        test_transforms = transforms.Compose([])
        test_imagefolder = GrayscaleImageFolder('../data_test', test_transforms)
        test_transforms = torch.utils.data.DataLoader(test_imagefolder, batch_size=1, shuffle=True)

        os.makedirs('test_recolor/gray/', exist_ok=True)
        os.makedirs('test_recolor/color/', exist_ok=True)

        for i, (input_gray, input_ab, target) in enumerate(test_transforms):
            use_gpu = True
            if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

            input_ab = class2ab(ab2class(input_ab))
            for j in range(min(len(input_gray), 5)):  # save at most 5 images
                save_path = {'grayscale': 'test_recolor/gray/', 'colorized': 'test_recolor/color/'}
                save_name = "img-seen2-{}.jpg".format(i)
                to_rgb(input_gray[j].cpu(), input_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

        return 0

    def test_colorize_from_model (self):
        test_transforms = transforms.Compose([])
        test_imagefolder = GrayscaleImageFolder('../data_train', test_transforms)
        test_transforms = torch.utils.data.DataLoader(test_imagefolder, batch_size=1, shuffle=True)

        os.makedirs('test_model/gray/', exist_ok=True)
        os.makedirs('test_model/color/', exist_ok=True)

        model = Cu_net()
        model.load_state_dict(torch.load('outputs_5/model-epoch-13-losses-3.952.pth'))
        model.eval()

        for i, (input_gray, input_ab, target) in enumerate(test_transforms):
            # use_gpu = True
            # if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

            # input_gray = torch.Tensor(input_gray, )
            output = model(input_gray)
            output = class2ab(prob2class(output))

            for j in range(min(len(input_gray), 5)):  # save at most 5 images
                save_path = {'grayscale': 'test_model/gray/', 'colorized': 'test_model/color/'}
                save_name = "img-seen-{}.jpg".format(i)
                to_rgb(input_gray[j].cpu(), output[j].detach().cpu(), save_path=save_path, save_name=save_name)

        return 0




