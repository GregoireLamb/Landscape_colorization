import math
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms

class2mapping = torch.tensor([
    25, 26, 27, 40, 41, 42, 51, 52, 53, 54,55, 56, 57, 64, 65, 66, 67, 68, 69, 70,71, 72, 79, 80, 81, 82, 83, 84,
    85, 86,87, 93, 94, 95, 96, 97, 98, 99, 100, 101,102, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 123, 124, 125, 126, 127, 128, 129, 130, 131,132, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 146, 147, 150, 151, 152, 153, 154, 155, 156,157, 158, 159, 160, 165, 166, 167, 168, 169, 170,
    171, 172, 173, 174, 175, 180, 182, 183, 184, 185,186, 187, 188, 189, 190
], dtype=torch.long).cuda()

mapping2class = torch.full((225,), 104, dtype=torch.long).long().cuda()
mapping2class[class2mapping] = torch.arange(len(class2mapping)).long().cuda()

def class2ab(te, n_classes=104):
    """
    :param te: torch.Size([1, 256, 256]): batch_size, height, width (value = classes)
    :return: torch.Size([1, 2, 256, 256]): batch_size, ab, height, width
    """
    ab = torch.zeros(te.shape[0], 2, te.shape[1], te.shape[2], dtype=torch.float32)

    if n_classes == 104:
        where = torch.where(te == 106)  # 106 is the error class make a and b  0
        te[where] = 104
        te = class2mapping[te]
        a_channel, b_channel = ((te // 15) / 15 + 1/30, (te % 15) / 15 + 1/30)
    elif n_classes == 100:
        a_channel, b_channel = ((te // 10) / 10 + 0.05, (te % 10) / 10 + 0.05)

    # Assign values to the corresponding channels in the output tensor
    ab[:, 0, :, :] = a_channel
    ab[:, 1, :, :] = b_channel
    return ab

def ab2prob(te, n_classes=104):
    """
    :param te: torch.Size([1, 2, 256, 256]): batch_size, ab, height, width
    :return: torch.Size([1, 100, 256, 256]): batch_size, Q_truth, height, width
    1/0 for now
    """
    batch_size, _, height, width = te.shape
    prob = torch.zeros(batch_size, n_classes, height, width).cuda()
    prob[torch.arange(batch_size).unsqueeze(1), ab2class(te, n_classes=n_classes), torch.arange(height).unsqueeze(0).unsqueeze(2), torch.arange(width).unsqueeze(0).unsqueeze(1)] = 1
    return prob

def ab2class(ab, n_classes=104):
    """
    :param ab: values in [0,1]
    :param n_classes: default 313
    :return: class [0,n_classes-1]
    """
    if n_classes == 104:
        mapping = (torch.round(ab[:, 0, :, :]*14)*15 + torch.round(ab[:, 1, :, :]*14)).clone().long() # class in a 15*15 grid
        return mapping2class[mapping]
    elif n_classes == 100:
        return (torch.floor(ab[:, 0, :, :] * 10) * 10 + torch.floor(ab[:, 1, :, :]*10)).clone().long()
    else:
        raise NotImplementedError

def prob2ab(te, n_classes=104, temperature=1, strategy="prob_max"):
    """
    :param tensor: te: torch.Size([1, 100, 256, 256]): batch_size, Q, height, width (value = prob, [0,1]
    :return: ab: torch.Size([1, 2, 256, 256]): batch_size, ab, height, width (value = classes)
    """
    te = te.cuda()
    batch_size, _, height, width = te.shape

    if strategy == "prob_max":
        return class2ab(torch.argmax(te, dim=1), n_classes=n_classes)
    elif strategy == "prob_max_temperature":
        raise  NotImplementedError
    elif strategy == "rebalanced_mean_prob":
        # Sum along the Q dimension and divide by the sum of the input tensor along the Q dimension
        ab_mean = prob2ab(te, n_classes=n_classes, strategy="mean_prob").cuda()
        ab_max = prob2ab(te, n_classes=n_classes, strategy="prob_max").cuda()*temperature
        ab = (ab_mean+ab_max)/(1+temperature)
        return ab
    elif strategy == "mean_prob":
        # Sum along the Q dimension and divide by the sum of the input tensor along the Q dimension
        ab = torch.zeros(batch_size, 2, height, width, dtype=torch.float32).cuda()
        ab_all_q = torch.zeros(batch_size, 225, height, width, dtype=torch.float32).cuda()
        if n_classes == 104:

            # from class [0-104] to class [0-224]
            for i in range(105):
                ab_all_q[:, class2mapping[i]] = te[:, i, :, :]

            multiplication_factors_a1 = torch.tensor([(i) // 15 * 1 / 15 + 1 / 30 for i in range(225)]).cuda()
            multiplication_factors_b1 = torch.tensor([(i) % 15 * 1 / 15 + 1 / 30 for i in range(225)]).cuda()
            multiplication_factors_a = torch.zeros(batch_size, 225, height, width, dtype=torch.float32).cuda()
            multiplication_factors_b = torch.zeros(batch_size, 225, height, width, dtype=torch.float32).cuda()

            for q, factor_a1, factor_b1 in zip(range(len(multiplication_factors_a1)), multiplication_factors_a1,
                                               multiplication_factors_b1):
                multiplication_factors_a[:, q, :, :] = factor_a1
                multiplication_factors_b[:, q, :, :] = factor_b1

            # from class [0-224] to a/b
            a_mult = ab_all_q * multiplication_factors_a
            b_mult = ab_all_q * multiplication_factors_b
            te = te.cuda()
            ab[:, 0, :, :] = torch.sum(a_mult, dim=1) / torch.exp(torch.log(torch.sum(te, dim=1)/temperature))
            ab[:, 1, :, :] = torch.sum(b_mult, dim=1) / torch.exp(torch.log(torch.sum(te, dim=1)/temperature))

            return ab

def process_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist

    # Loop through each file in the input folder
    for folder in os.listdir(input_folder):# Test train and validation
        if not os.path.exists(os.path.join(output_folder, folder)):
            os.makedirs(os.path.join(output_folder, folder))
        for cat_image in os.listdir(os.path.join(input_folder, folder)):# Coast Desert Forest Glacier Mountain
            if not os.path.exists(os.path.join(output_folder, folder, cat_image)):
                os.makedirs(os.path.join(output_folder, folder, cat_image))
            for image in os.listdir(os.path.join(input_folder, folder, cat_image)):
                img = Image.open(os.path.join(input_folder, folder, cat_image, image))

                # Crop the image to a square
                width, height = img.size
                size = min(width, height)
                left = (width - size) // 2
                top = (height - size) // 2
                right = (width + size) // 2
                bottom = (height + size) // 2
                img = img.crop((left, top, right, bottom))

                # Resize the image to 256x256 pixels
                img = img.resize((256, 256), Image.ANTIALIAS)

                # Save the processed image to the output folder
                output_path = os.path.join(output_folder, folder, cat_image, image)
                img.save(output_path, quality=85)  # You can adjust the quality parameter as needed

def plot_loss_evolution(losses, save_path):
    plt.plot(losses)
    plt.title("Loss evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(save_path)
    plt.close()

def compute_euclidean_distance_2_images(image1, image2):
    """
    Compute the euclidean distance between two images
    :param image1: torch.Tensor
    :param image2: torch.Tensor
    :return: torch.Tensor
    """
    return torch.sum(torch.sqrt(torch.sum((image1 - image2) ** 2, dim=1))).item()

def compute_PSNR_2_images(image1, image2):
    return 10 * math.log(255*255/compute_euclidean_distance_2_images(image1, image2), 10)

def compute_distances_metric(path_folder1, path_folder2, metric="euclidean"):
    """
    Compute the euclidean distance between two set of images
    :param image1: torch.Tensor
    :param image2: torch.Tensor
    :return: torch.Tensor
    """
    distances = []
    # load the images
    for im_truth in os.listdir(path_folder1):
        for im_pred in os.listdir(path_folder2):
            # if im_truth has the same name as im_pred
            if im_truth.split('.')[0] == im_pred.split('.')[0]:
                im_truth = Image.open(os.path.join(path_folder1, im_truth))
                im_pred = Image.open(os.path.join(path_folder2, im_pred))
                if metric == "euclidean":
                    distances.append(compute_euclidean_distance_2_images(transforms.ToTensor()(im_truth), transforms.ToTensor()(im_pred)))
                    break
                elif metric == "PSNR":
                    distances.append(compute_PSNR_2_images(transforms.ToTensor()(im_truth), transforms.ToTensor()(im_pred)))
                    break
                elif metric == "Accuracy":
                    print("WARNING: metric not implemented")
                    break
                else:
                    print("WARNING: metric not implemented")
            # Add message no image found
    if len(os.listdir(path_folder1)) != len(os.listdir(path_folder2)) or len(os.listdir(path_folder1)) != len(distances):
        print("WARNING: not the same number of images in both folders or image haven't the same names")
        print("Folder1:", len(os.listdir(path_folder1)))
        print("Folder2:", len(os.listdir(path_folder2)))
        print("Common images:", len(distances))
    return distances

def load_checkpoint(model, optimizer, filename, device):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    im_to_restart_from = 0
    if os.path.isfile(filename):
        ckpt = torch.load(filename, map_location = device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

        print("loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'i' in checkpoint.keys():
            im_to_restart_from = checkpoint['i']
        print("Loaded checkpoint '{}' 100% \n".format(filename))
    else:
        print("No checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, im_to_restart_from

def data_splitter(path):
    """
    :param path: path to the dataset
    :return: train, test, validation
    """
    # set seed for reproducibility
    torch.manual_seed(123)

    train_path = path+'/data_train/images'
    test_path = path+'/data_test/images'
    validation_path = path+'/data_validation/images'

    # create the folders
    os.makedirs(validation_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)

    # load the images and split in train, test, validation 75% 10% 15%
    counter = 0
    for image in os.listdir(path+"/data_train/images"):
        if counter%1000 == 0:
            print(counter/900, "%")
        counter += 1
        # generate a random number
        rand = torch.rand(1).item()
        if rand < 0.1:
            # move the image to validation
            os.rename(train_path+'/'+image, test_path+'/'+image)
        elif rand < 0.25:
            os.rename(train_path+'/'+image, validation_path+'/'+image)

def data_split_img_in_2(path):
    """
    :param path: path to the dataset
    :return: train, test, validation
    """
    images1 = path+'/images'
    images2 = path+'/images2'

    # create the folders
    os.makedirs(images1, exist_ok=True)
    os.makedirs(images2, exist_ok=True)

    # load the images and split in train, test, validation 75% 10% 15%
    counter = 0
    for image in os.listdir(images1):
        if counter%2 == 0:
            os.rename(images1+'/'+image, images2+'/'+image)
        if counter%1000 == 0:
            print(counter)
        counter += 1

def gaussian_tens(tens, sig = 0.1):
    coefficient = 1 / (math.sqrt(2 * 3.1416) * sig)
    exponent = -(tens ** 2) / (2 * sig ** 2)
    return coefficient * torch.exp(exponent) / 2

def gaussian(a,b, x, y, sig = 0.1):
    """
    :param a:  a truth
    :param b: b truth
    :param x: a_class
    :param y: b_class
    :param sig: gaussian sigma
    :return: distance between a,b and x,y
    """
    dist = torch.sqrt((a - x) ** 2 + (b - y) ** 2) # euclidean distance
    coefficient = 1 / (math.sqrt(2 * math.pi) * sig)
    exponent = -(dist ** 2) / (2 * sig ** 2)
    return coefficient * torch.exp(exponent)/2

def un_gaussian(dist, sig=0.1):
    """
    :param dist: distance between a,b and x,y (gaussian)
    :param sig: gaussian sigma
    :return: euclidean distance
    """
    coefficient = 1 / (math.sqrt(2 * math.pi) * sig)
    non_null_mask = (dist != 0)  # Create a mask for non-null values
    non_null_indices = non_null_mask.nonzero(as_tuple=False).squeeze(1)#.unsqueeze(1) # Get indices of non-null values
    non_null_dist = dist[non_null_mask]  # Apply mask to obtain non-null distances

    edist = torch.sqrt(-2 * sig ** 2 * torch.log(non_null_dist / coefficient) / torch.log(torch.tensor(2.7182818284)))
    euclidean_distance = torch.zeros_like(dist)

    euclidean_distance[non_null_indices[:, 0],
                        non_null_indices[:, 1],
                        non_null_indices[:, 2],
                        non_null_indices[:, 3]] = edist

    return euclidean_distance
