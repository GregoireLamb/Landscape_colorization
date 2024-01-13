import os
import shutil

from skimage.color import lab2rgb

from src.GrayscaleImageFolder import *
from src.Cu_net import *
from src.util import *

def colorize(path):
    """
    Generate the 5 images for the given image
    """
    os.makedirs(f'./images/provided/img/', exist_ok=True)
    os.makedirs(f'./images/gray/', exist_ok=True)
    os.makedirs(f'./images/model_1/', exist_ok=True)
    os.makedirs(f'./images/model_2/', exist_ok=True)
    os.makedirs(f'./images/model_3/', exist_ok=True)

    # Empty the folder ./images/provided/img/
    for filename in os.listdir('./images/provided/img/'):
        file_path = os.path.join('./images/provided/img/', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    image_to_colorize_path = './images/provided/'
    # copy the image to colorize in the folder
    shutil.copy(path, './images/provided/img/')

    demo_transform = transforms.Compose([])
    demo_imagefolder = GrayscaleImageFolder(image_to_colorize_path, demo_transform)
    demo_transform = torch.utils.data.DataLoader(demo_imagefolder, batch_size=1, shuffle=False)

    save_gray = True

    for model_path, temp in [('./models/model_1.pth', 2), ('./models/model_2.pth', 0.25), ('./models/model_3.pth',2)]:
        model = Cu_net()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        for i, (input_gray, input_ab, target) in enumerate(demo_transform):
            pred = model(input_gray)
            output = prob2ab(pred, n_classes=104, strategy="rebalanced_mean_prob", temperature=temp)
            save_name = path.split('/')[-1].split('.')[0] + "_model_" + model_path[-5]+str(i)

            for j in range(len(input_gray)):
                output = to_rgb(input_gray[j].cpu(), output[j].detach().cpu(), save_path=f'./images/{model_path[9:16]}/', save_name=save_name)
                if save_gray:
                    to_rgb(input_gray[j].cpu(), 0, save_path=f'./images/gray/', save_name=save_name, save_gray=save_gray)
        save_gray = False
    return

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

