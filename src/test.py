import os
from unittest import TestCase

import numpy as np
import torch

from src.Cu_net_small import Cu_net_small
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
        tensor = torch.tensor([[[[0.0, 0.09],
                                 [0.5, 1]],
                                [[0.0, 0.72],
                                 [0.5, 1]]]]) # torch.Size([1, 2, 2, 2])
        classes = ab2class(tensor, n_classes=105)
        assert(classes[0, 0, 0] == 106)
        assert(classes[0, 0, 1] == 0)
        assert(classes[0, 1, 0] == 45)
        assert(classes[0, 1, 1] == 106)

        return 0

    def test_class2ab(self):
        classes = torch.tensor([[[0, 97],
                                 [106, 106]]])  # torch.Size([1, 2, 2])
        tensor = class2ab(classes, n_classes=105)
        #A
        assert(np.round(tensor[0, 0, 0, 0].item(), decimals=3) == .1)
        assert(np.round(tensor[0, 0, 0, 1].item(), decimals=3) == .833)
        assert(np.round(tensor[0, 0, 1, 0].item(), decimals=3) == 0.833)
        assert(np.round(tensor[0, 0, 1, 1].item(), decimals=3) == 0.833)
        #B
        assert (np.round(tensor[0, 1, 0, 0].item(), decimals=3) == .7)
        assert (np.round(tensor[0, 1, 0, 1].item(), decimals=3) == .233)
        assert (np.round(tensor[0, 1, 1, 0].item(), decimals=3) == 0.700)
        assert (np.round(tensor[0, 1, 1, 1].item(), decimals=3) == 0.700)

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
        image = class2ab(ab2class(all_ab, n_classes=105),  n_classes=105).squeeze(0)
        for j in range(min(len(image), 5)):
            save_path = {'grayscale': 'test_color/gray/', 'colorized': 'test_color/color/'}
            save_name = 'img_105_color-L-{}.jpg'.format(L)
            image = image.detach()
            to_rgb(teL.cpu(), ab_input=image.detach().cpu(), save_path=save_path, save_name=save_name)
        return 0
    #
    # def test_all_classes (self):
    #     os.makedirs('test_color/gray/', exist_ok=True)
    #     os.makedirs('test_color/color/', exist_ok=True)
    #
    #     teL = torch.zeros(1, 256, 256, dtype=torch.float32)
    #     classes = torch.zeros(1, 256, 256, dtype=torch.float32)
    #     image = torch.zeros(2, 256, 256, dtype=torch.float32)
    #     for i in range(0, 256):
    #         for j in range(0, 256):
    #             classes[0, i, j] = int(i/256 *100)
    #             teL[0, i, j] = j/256
    #
    #     for i in range(0, 256):
    #         for j in range(0, 256):
    #             image[0, i, j], image[1, i, j] = class2ab(classes[0,i,j])
    #
    #     for j in range(min(len(image), 5)):
    #         save_path = {'grayscale': 'test_color/gray/', 'colorized': 'test_color/color/'}
    #         save_name = 'img2-{}-epoch-{}.jpg'.format(43,50)
    #         image = image.detach()
    #         to_rgb(teL.cpu(), ab_input=image.detach().cpu(), save_path=save_path, save_name=save_name)
    #     return 0
    #
    def test_recolorize (self):
        test_transforms = transforms.Compose([])
        test_imagefolder = GrayscaleImageFolder('../data_test', test_transforms)
        test_transforms = torch.utils.data.DataLoader(test_imagefolder, batch_size=1, shuffle=True)

        os.makedirs('test_recolor/gray/', exist_ok=True)
        os.makedirs('test_recolor/color/', exist_ok=True)

        predicted_colors = set()

        for i, (input_gray, input_ab, target) in enumerate(test_transforms):
            use_gpu = True
            if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

            prob = ab2prob(input_ab, n_classes=105)
            input_ab = prob2ab(prob, temperature=1, n_classes=105)
            for cl in torch.unique(input_ab):
                predicted_colors.add(cl.item())

            for j in range(min(len(input_gray), 5)):  # save at most 5 images
                save_path = {'grayscale': 'test_recolor/gray/', 'colorized': 'test_recolor/color/'}
                save_name = "img105colors-{}.jpg".format(i)
                to_rgb(input_gray[j].cpu(), input_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)
        print("len(predicted_colors)", len(predicted_colors))
        print(len(predicted_colors))
        return 0

    def test_recolorize_no_class (self):
        test_transforms = transforms.Compose([])
        test_imagefolder = GrayscaleImageFolder('../data_train_1', test_transforms)
        test_transforms = torch.utils.data.DataLoader(test_imagefolder, batch_size=1, shuffle=True)

        os.makedirs('test_recolor/gray/', exist_ok=True)
        os.makedirs('test_recolor/color/', exist_ok=True)

        predicted_colors = set()

        for i, (input_gray, input_ab, target) in enumerate(test_transforms):
            use_gpu = True
            if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

            for cl in torch.unique(input_ab):
                predicted_colors.add(cl.item())

            for j in range(min(len(input_gray), 5)):  # save at most 5 images
                save_path = {'grayscale': 'test_recolor/gray/', 'colorized': 'test_recolor/color/'}
                save_name = "img_recolor_no_class-{}.jpg".format(i)
                to_rgb(input_gray[j].cpu(), input_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)
        print("len(predicted_colors)", len(predicted_colors))
        print(len(predicted_colors))
        return 0

    def test_colorize_from_model (self):
        test_transforms = transforms.Compose([])
        test_imagefolder = GrayscaleImageFolder('../data_test', test_transforms)
        test_transforms = torch.utils.data.DataLoader(test_imagefolder, batch_size=1, shuffle=True)

        os.makedirs('test_model/cu_medium_ep64/gray/', exist_ok=True)
        os.makedirs('test_model/cu_medium_ep64/color/', exist_ok=True)

        model = Cu_net()
        model.load_state_dict(torch.load('../src/checkpoints/cu_medium-epoch-64-losses-3.885.pth'))
        model.eval()

        for i, (input_gray, input_ab, target) in enumerate(test_transforms):

            output = prob2ab(model(input_gray), n_classes=105)

            for j in range(len(input_gray)):  # save at most 5 images
                save_path = {'grayscale': 'test_model/cu_medium_ep64/gray/', 'colorized': 'test_model/cu_medium_ep64/color/'}
                save_name = "img-{}.jpg".format(i)
                to_rgb(input_gray[j].cpu(), output[j].detach().cpu(), save_path=save_path, save_name=save_name)

        return 0
    #
    # def test_gaussian(self):
    #     sig = 0.2
    #     # plot a function chart of the gaussian function
    #     val = np.zeros(100)
    #     for dist in range(0, 100):
    #         coefficient = 1 / (math.sqrt(2 * math.pi) * sig)
    #         exponent = -((dist/100*0.15)** 2) / (2 * sig ** 2)
    #         print(f"{(dist/100*0.15)}")
    #         val[dist] = coefficient * math.exp(exponent)/2
    #
    #     plt.plot(val)
    #     #plot y axis from 0 to 1
    #     # plt.ylim(0, 0.04)
    #     plt.show()
    #     return 0
    #
    #
    # def test_ab2prob2(self): # seems ok
    #     te = torch.tensor([[[[0.05, 0],
    #                          [0, 0]],
    #                          [[0.15, 0],
    #                          [0, 0]]]], dtype=torch.float32).cuda()
    #     print(te.shape)
    #     prob = ab2prob(te)
    #     print(prob.shape)
    #     print(prob)
    #
    #     assert(prob[0, 0, 0, 0] == 0)
    #     assert(prob[0, 0, 0, 1] == 1)
    #     assert(prob[0, 1, 0, 0] == 1)
    #     return 0
    #
    def test_behaviour_bn(self):
        test_transforms = transforms.Compose([])
        test_imagefolder = GrayscaleImageFolder('../data_train_1', test_transforms)
        test_transforms = torch.utils.data.DataLoader(test_imagefolder, batch_size=1, shuffle=True)

        for i, (input_gray, input_ab, target) in enumerate(test_transforms):
            use_gpu = True
            if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()
            print(input_gray.shape)
            print(input_ab.shape)
            print(input_ab)
            print(target.shape)
            return 0
        return 0