from unittest import TestCase
import os
import numpy as np
import torch

from src.Cu_net_small import Cu_net_small
from src.GrayscaleImageFolder import GrayscaleImageFolder
from src.main import to_rgb, get_class_penalty
from src.util import *

import os
import sys

from src.Cu_net import *
from src.GrayscaleImageFolder import *
from src.util import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Test(TestCase):
    def test_ab2class(self):
        tensor = torch.tensor([[[[0.0, 0.09],
                                 [0.5, 1]],
                                [[0.0, 0.72],
                                 [0.5, 1]]]]) # torch.Size([1, 2, 2, 2])
        classes = ab2class(tensor, n_classes=105)
        print(classes)
        assert(classes[0, 0, 0] == 104)
        assert(classes[0, 0, 1] == 0)
        assert(classes[0, 1, 0] == 45)
        assert(classes[0, 1, 1] == 104)

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
        prob = ab2prob(tensor, n_classes=100)
        assert(len(prob.shape) == 4)
        assert(prob[0, 66, 1, 1] == 1)
        assert(prob[0, 55, 1, 1] == 0)
        assert(prob[0, 24, 0, 1] == 1)
        assert(tensor != ab2prob(tensor, n_classes=105))
        return 0

    def test_all_colors (self):
        L = 80
        os.makedirs('test_color/gray/', exist_ok=True)
        os.makedirs('test_color/color/', exist_ok=True)
        teL = torch.zeros(1, 256, 256, dtype=torch.float32)
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

    def test_recolorize (self):
        test_transforms = transforms.Compose([])
        test_imagefolder = GrayscaleImageFolder('../data_small/data_test/', test_transforms)
        test_transforms = torch.utils.data.DataLoader(test_imagefolder, batch_size=1, shuffle=False)

        os.makedirs('test_recolor/gray/', exist_ok=True)
        os.makedirs('test_recolor/color_100_classes/', exist_ok=True)
        os.makedirs('test_recolor/color_104_classes/', exist_ok=True)

        predicted_colors = set()

        for n_c in [100, 104]:
            for i, (input_gray, input_ab, target) in enumerate(test_transforms):
                use_gpu = True
                if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

                prob = ab2prob(input_ab, n_classes=n_c)
                input_ab = prob2ab(prob, temperature=1, n_classes=n_c, strategy="prob_max")
                for cl in torch.unique(input_ab):
                    predicted_colors.add(cl.item())

                for j in range(min(len(input_gray), 5)):  # save at most 5 images
                    save_path = {'grayscale': 'test_recolor/gray/', 'colorized': f'test_recolor/color_{n_c}_classes/'}
                    save_name = f'img_{n_c}_colors_{i}.jpg'
                    to_rgb(input_gray[j].cpu(), input_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)
            print("Used ", len(predicted_colors)," different colors among ", n_c, " classes")

        return 0

    def test_recolorize_no_class (self):
        """
        Recolorize images without using classes, the output should be the exact same as the input
        """
        os.makedirs('test_recolor/gray/', exist_ok=True)
        os.makedirs('test_recolor/no_classes/', exist_ok=True)

        test_transforms = transforms.Compose([])
        test_imagefolder = GrayscaleImageFolder('../data_small/data_test/', test_transforms)
        test_transforms = torch.utils.data.DataLoader(test_imagefolder, batch_size=1, shuffle=False)

        predicted_colors = set()

        for i, (input_gray, input_ab, target) in enumerate(test_transforms):
            use_gpu = True
            if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

            for cl in torch.unique(input_ab):
                predicted_colors.add(cl.item())

            for j in range(min(len(input_gray), 5)):  # save at most 5 images
                save_path = {'grayscale': 'test_recolor/gray/', 'colorized': 'test_recolor/no_classes/'}
                save_name = "img_recolor_no_class-{}.jpg".format(i)
                to_rgb(input_gray[j].cpu(), input_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)
        print("len(predicted_colors)", len(predicted_colors), " for ", len(test_transforms) , " images")
        return 0

    def test_colorize_from_model (self):
        rebalance_factor = 0.2 # 0 is no effect
        test_transforms = transforms.Compose([])
        test_imagefolder = GrayscaleImageFolder('../data_small/data_test', test_transforms)
        test_transforms = torch.utils.data.DataLoader(test_imagefolder, batch_size=1, shuffle=False)

        model = Cu_net()
        model_name = 'epoch-3.pth'
        ckpt = torch.load('C:/Users/gdela/Downloads/epoch-8.pth')
        model.load_state_dict(ckpt['state_dict'])
        # model.load_state_dict(torch.load('./checkpoints/no_penalty_ep15_3.8836.pth'))
        model.eval()

        os.makedirs(f'test_model/{model_name}/gray/', exist_ok=True)
        os.makedirs(f'test_model/{model_name}/color/', exist_ok=True)
        os.makedirs(f'test_model/{model_name}/truth/', exist_ok=True)

        penalty_105 = 1 / get_class_penalty(use_precompute=True)
        penalty_105_expanded = penalty_105.view(1, 105, 1, 1)
        penalty = penalty_105_expanded.expand(1, 105, 256, 256)
        penalty = penalty.cuda()

        for i, (input_gray, input_ab, target) in enumerate(test_transforms):
            pred = model(input_gray)
            pred = pred.cuda()

            pred = pred - rebalance_factor * penalty
            output = prob2ab(pred, n_classes=104, strategy="rebalanced_mean_prob", temperature=0.2)

            for j in range(len(input_gray)):  # save at most 5 images
                save_path = {'grayscale': f'test_model/{model_name}/gray/',
                             'colorized': f'test_model/{model_name}/color/',
                             'truth': f'test_model/{model_name}/truth/'}
                save_name = "img-{}.jpg".format(i)
                to_rgb(input_gray[j].cpu(), output[j].detach().cpu(), save_path=save_path, save_name=save_name)
                to_rgb(input_gray[j].cpu(), input_ab[j].detach().cpu(), save_path=save_path, save_name=save_name, truth=True)

        return 0

    def test_compute_euclidean_distance_2_same_images(self):
        img = Image.open('./test_model/epoch-4.pth/color/img-0.jpg')
        img2 = Image.open('./test_model/epoch-4.pth/truth/img-0.jpg')
        img = transforms.ToTensor()(img)
        img2 = transforms.ToTensor()(img2)

        assert compute_euclidean_distance_2_images(img, img) == 0
        assert compute_euclidean_distance_2_images(img, img2) != 0
        return 0

    def test_compute_metric_distances(self):
        print("--- euclidean ")
        dists = compute_distances_metric('./test_model/epoch-4.pth/color/', './test_model/epoch-4.pth/truth/', metric="euclidean")
        print("color ", np.mean(dists))
        dists2 = compute_distances_metric('./test_model/epoch-4.pth/gray/', './test_model/epoch-4.pth/truth/', metric="euclidean")
        print("gray ", np.mean(dists2))
        print("--- PSNR ")
        dists = compute_distances_metric('./test_model/epoch-4.pth/color/', './test_model/epoch-4.pth/truth/', metric="PSNR")
        print("color ",np.mean(dists))
        dists2 = compute_distances_metric('./test_model/epoch-4.pth/gray/', './test_model/epoch-4.pth/truth/', metric="PSNR")
        print("gray ", np.mean(dists2))
        return 0

    def test_temperature(self):
        torch.manual_seed(123) #car
        # torch.manual_seed(14125456)
        temp = [0, 0.25, 0.4, 0.5, 0.75, 1, 3, 25]
        test_transforms = transforms.Compose([])
        test_imagefolder = GrayscaleImageFolder('../data/data_test', test_transforms)
        test_transforms = torch.utils.data.DataLoader(test_imagefolder, batch_size=1, shuffle=True)

        os.makedirs('test_model/temperature/gray/', exist_ok=True)
        os.makedirs('test_model/temperature/color/', exist_ok=True)
        os.makedirs('test_model/temperature/truth/', exist_ok=True)

        model = Cu_net()

        ckpt = torch.load('C:/Users/gdela/Downloads/epoch-0_img-450.pth')
        model.load_state_dict(ckpt['state_dict'])
        # model.load_state_dict(torch.load('./checkpoints/no_penalty_ep15_3.8836.pth'))
        model.eval()

        print("model loaded")
        for i, (input_gray, input_ab, target) in enumerate(test_transforms):
            print("i", i)
            pred = model(input_gray)
            count_t = 0
            for t in temp:
                output = prob2ab(pred, n_classes=104, strategy="rebalanced_mean_prob", temperature=t)
                save_path = {'grayscale': 'test_model/temperature/gray/',
                             'colorized': 'test_model/temperature/color/',
                             'truth': 'test_model/temperature/truth/'}
                save_name = f"img-{i}_{count_t}_temp-{t}.jpg"
                count_t += 1
                to_rgb(input_gray[0].cpu(), output[0].detach().cpu(), save_path=save_path, save_name=save_name)
                to_rgb(input_gray[0].cpu(), input_ab[0].detach().cpu(), save_path=save_path, save_name=save_name,
                       truth=True)
            if i>10:
                break

        return 0
    def test_rebal_penalty(self):
        torch.manual_seed(123) #car
        # torch.manual_seed(14125456)
        factor = [0.25 * i for i in range(5)]
        test_transforms = transforms.Compose([])
        test_imagefolder = GrayscaleImageFolder('../data/data_test', test_transforms)
        test_transforms = torch.utils.data.DataLoader(test_imagefolder, batch_size=1, shuffle=True)

        os.makedirs('test_model/temperature/gray/', exist_ok=True)
        os.makedirs('test_model/temperature/color/', exist_ok=True)
        os.makedirs('test_model/temperature/truth/', exist_ok=True)

        model = Cu_net()

        ckpt = torch.load('C:/Users/gdela/Downloads/epoch-1.pth')
        model.load_state_dict(ckpt['state_dict'])
        # model.load_state_dict(torch.load('./checkpoints/no_penalty_ep15_3.8836.pth'))
        model.eval()
        print("model loaded")
        penalty_105 = 1/get_class_penalty(use_precompute=True)
        penalty_105_expanded = penalty_105.view(1, 105, 1, 1)
        penalty = penalty_105_expanded.expand(1, 105, 256, 256)
        penalty = penalty.cuda()

        for i, (input_gray, input_ab, target) in enumerate(test_transforms):
            print("i", i)
            pred = model(input_gray)
            pred = pred.cuda()
            count_t = 0
            for fact in factor:
                pred = pred - fact * penalty

                output = prob2ab(pred, n_classes=104, strategy="rebalanced_mean_prob", temperature=0.45)
                save_path = {'grayscale': 'test_model/temperature/gray/',
                             'colorized': 'test_model/temperature/color/',
                             'truth': 'test_model/temperature/truth/'}
                save_name = f"img-{i}_{count_t}_temp-{fact}.jpg"
                count_t += 1
                to_rgb(input_gray[0].cpu(), output[0].detach().cpu(), save_path=save_path, save_name=save_name)
                to_rgb(input_gray[0].cpu(), input_ab[0].detach().cpu(), save_path=save_path, save_name=save_name,
                       truth=True)
            if i>10:
                break

        return 0
