import os

from skimage.color import lab2rgb

from src.GrayscaleImageFolder import *
from src.Cu_net import *
from src.util import *


def colorize(path):
    path = path.replace('{', '')
    path = path.replace('}', '')
    # Assert path is a directory
    print("Path: ", path)
    if os.path.isdir(path):
        colorize(path)
    else:
        print("Path is not a directory")
    return 0


def colorize(path):
    """
    Generate the 5 images for the given image
    """
    os.makedirs(f'{path}/gray/', exist_ok=True)
    os.makedirs(f'{path}/model_1/', exist_ok=True)
    os.makedirs(f'{path}/model_2/', exist_ok=True)
    os.makedirs(f'{path}/model_3/', exist_ok=True)

    demo_transform = transforms.Compose([])
    demo_imagefolder = GrayscaleImageFolder(path, demo_transform)
    demo_transform = torch.utils.data.DataLoader(demo_imagefolder, batch_size=1, shuffle=False)

    images = []
    save_gray = True

    for model_path, temp in [('./models/model_1.pth', 2), ('./models/model_2.pth', 0.25), ('./models/model_3.pth',2)]:
        model = Cu_net()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        for i, (input_gray, input_ab, target) in enumerate(demo_transform):
            pred = model(input_gray)
            output = prob2ab(pred, n_classes=104, strategy="rebalanced_mean_prob", temperature=temp)
            save_name = path.split('/')[-1] + "_model_" + model_path[-5]+str(i)

            for j in range(len(input_gray)):
                output = to_rgb(input_gray[j].cpu(), output[j].detach().cpu(), save_path=f'{path}/{model_path[9:16]}/', save_name=save_name)
                if save_gray:
                    to_rgb(input_gray[j].cpu(), 0, save_path=f'{path}/gray/', save_name=save_name, save_gray=save_gray)
                images.append(output)
        save_gray = False
    return images

def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None, save_gray=False):
    '''Show/save rgb image from grayscale and ab channels
       Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    if save_gray:
        grayscale_input = grayscale_input.squeeze().numpy()
        plt.imsave(arr=grayscale_input, fname=save_path+"/"+save_name+".jpg", cmap='gray')
        return grayscale_input
    plt.clf()  # clear matplotlib
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy()  # combine channels
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    color_image = lab2rgb(color_image.astype(np.float64))

    plt.imsave(arr=color_image, fname=save_path+"/"+save_name+".jpg")
    return color_image
#
# if __name__ == "__main__":
#     colorize("C:/Users/gdela/OneDrive/Images/demo_images")

