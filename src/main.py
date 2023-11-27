import os
import sys

from alive_progress import alive_bar
from skimage.color import xyz2rgb
from skimage.color.colorconv import _prepare_colorarray, get_xyz_coords

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Cu_net import *
from src.U_net_small import *
from src.Cu_net_small import *
from src.GrayscaleImageFolder import *
from src.util import *
import click

def main():
    click.clear()

    # Check if GPU is available
    use_gpu = torch.cuda.is_available()
    model = Cu_net()
    # model = U_net_small()
    # model = Cu_net_small()
    n_classes = 105
    epochs = 64
    batch_size = 32
    criterion = nn.CrossEntropyLoss()
    lr = 1.5e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    save_images =True
    best_losses = 1e10
    T = 1 # temperature

    print("[LANDSCAPE COLORIZATION]\n")
    print("Parameters:")
    print("\tModel: {}".format(model.name))
    print("\tn_classes: {}".format(n_classes))
    print("\tUsing GPU: {}".format(use_gpu))
    print("\tEpochs: {}".format(epochs))
    print("\tBatch size: {}".format(batch_size))
    print("\tCriterion: {}".format(criterion))
    print("\tOptimizer: {}, learningrate: {}\n".format(optimizer.__class__.__name__, lr))


    train_transforms = transforms.Compose([])
    train_imagefolder = GrayscaleImageFolder('../data_train', train_transforms)
    train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=batch_size, shuffle=True)

    val_transforms = transforms.Compose([])
    val_imagefolder = GrayscaleImageFolder('../data_validation', val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=batch_size, shuffle=False)

    if use_gpu:
        criterion = criterion.cuda()
        model = model.cuda()

    # Make folders and set parameters
    os.makedirs('outputs/color', exist_ok=True)
    os.makedirs('outputs/gray', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # Train model
    print("Start training")
    for epoch in range(epochs):
        # Train for one epoch, then validate
        train(train_loader, model, criterion, optimizer, epoch, n_classes=n_classes)
        with torch.no_grad():
            losses = validate(val_loader, model, criterion, save_images, epoch, temperature=T, n_classes=n_classes)
        # Save checkpoint and store best model if current model is better
        if losses < best_losses:
            best_losses = losses
            torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch + 1, losses))


def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
  '''Show/save rgb image from grayscale and ab channels
     Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
  plt.clf() # clear matplotlib
  color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
  color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
  color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
  color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
  color_image = lab_to_rgb(color_image.astype(np.float64))

  grayscale_input = grayscale_input.squeeze().numpy()
  if save_path is not None and save_name is not None:
    plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
    plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))


def validate(val_loader, model, criterion, save_images, epoch, temperature, use_gpu=True, n_classes=105):
    model.eval()
    losses = 0
    count = 0
    already_saved_images = False
    with alive_bar(total=len(val_loader), title="Valid epoch: [{0}]".format(epoch),
                   spinner='classic') as bar:  # len(train_loader) = n_batches
        for i, (input_gray, input_ab, target) in enumerate(val_loader):
            if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

            # Run model and record loss
            output_prob = model(input_gray) # -> batch*256*256*100
            output_prob = torch.flatten(output_prob, start_dim=2).cuda()
            input_ab_class = torch.flatten(ab2class(input_ab, n_classes=n_classes), start_dim=1).long().cuda()
            loss = criterion(output_prob, input_ab_class)
            losses += loss.item()
            count += 1

            unflatten = torch.nn.Unflatten(2, (256, 256))# TODO adapt hard coded values
            output_prob = unflatten(output_prob)
            output_ab = ab2class(prob2ab(output_prob, n_classes=n_classes, temperature=temperature))
            output_ab = class2ab(output_ab, n_classes=n_classes)
            # Save images to file
            if save_images and not already_saved_images:
                already_saved_images = True
                for j in range(min(len(output_ab), 5)): # save at most 5 images
                    save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
                    save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
                    to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)
            bar()
    losses = float(losses)/count
    print("Loss {loss:.4f}\t".format(loss=losses))
    return losses


def train(train_loader, model, criterion, optimizer, epoch, use_gpu = True, save_images = True, n_classes=105):
    model.train()

    with alive_bar(total=len(train_loader), title="Train epoch: [{0}]".format(epoch), spinner='classic') as bar: #len(train_loader) = n_batches
        for i, (input_gray, input_ab, target) in enumerate(train_loader):
            if use_gpu: input_gray, input_ab, target= input_gray.cuda(), input_ab.cuda(), target.cuda()

            output_ab_class = model(input_gray)
            input_ab_class = ab2class(input_ab, n_classes=n_classes)

            if use_gpu: output_ab_class, input_ab_class, target = output_ab_class.cuda(), input_ab_class.cuda(), target.cuda()

            # desire shape is batch, Q, x for output and batch, x for input
            output_ab_class = torch.flatten(output_ab_class, start_dim=2)
            input_ab_class = torch.flatten(input_ab_class, start_dim=1).long()

            loss = criterion(output_ab_class,input_ab_class)

            # Compute gradient and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar()


def lab_to_rgb(lab, illuminant="D65", observer="2", *, channel_axis=-1):
    return xyz2rgb(custom_lab2xyz(lab, illuminant, observer))

def custom_lab2xyz(lab, illuminant="D65", observer="2", *, channel_axis=-1):
    arr = _prepare_colorarray(lab, channel_axis=-1).copy()

    L, a, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)

    if np.any(z < 0):
        invalid = np.nonzero(z < 0)
        warning('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)
        z[invalid] = 0
        x[invalid] = 0
        y[invalid] = 0

    out = np.stack([x, y, z], axis=-1)

    mask = out > 0.2068966
    out[mask] = np.power(out[mask], 3.)
    out[~mask] = (out[~mask] - 16.0 / 116.) / 7.787

    # rescale to the reference white (illuminant)
    xyz_ref_white = get_xyz_coords(illuminant, observer)
    out *= xyz_ref_white
    return out


def get_empirical_distribution(path_to_images="../data_sq", n_classes=105, cl=True):
    imagefolder = GrayscaleImageFolder(path_to_images, transforms.Compose([]))
    loader = torch.utils.data.DataLoader(imagefolder, batch_size=32, shuffle=True)

    class_count = torch.zeros(n_classes)
    for i, (_, inputab, _) in enumerate(loader):
        target = ab2class(inputab, n_classes=n_classes).flatten()

        class_count += torch.bincount(target.cpu(), minlength=n_classes)
        if i % 10 == 0:
            print("i", i)
        if i>100:
            break


    class2mapping = {
            0: 25, 1: 26, 2: 27, 3: 40, 4: 41, 5: 42, 6: 51, 7: 52, 8: 53, 9: 54,
            10: 55, 11: 56, 12: 57, 13: 64, 14: 65, 15: 66, 16: 67, 17: 68, 18: 69,
            19: 70, 20: 71, 21: 72, 22: 79, 23: 80, 24: 81, 25: 82, 26: 83, 27: 84,
            28: 85, 29: 86, 30: 87, 31: 93, 32: 94, 33: 95, 34: 96, 35: 97, 36: 98,
            37: 99, 38: 100, 39: 101, 40: 102, 41: 108, 42: 109, 43: 110, 44: 111,
            45: 112, 46: 113, 47: 114, 48: 115, 49: 116, 50: 117, 51: 123, 52: 124,
            53: 125, 54: 126, 55: 127, 56: 128, 57: 129, 58: 130, 59: 131, 60: 132,
            61: 136, 62: 137, 63: 138, 64: 139, 65: 140, 66: 141, 67: 142, 68: 143,
            69: 144, 70: 145, 71: 146, 72: 147, 73: 150, 74: 151, 75: 152, 76: 153,
            77: 154, 78: 155, 79: 156, 80: 157, 81: 158, 82: 159, 83: 160, 84: 165,
            85: 166, 86: 167, 87: 168, 88: 169, 89: 170, 90: 171, 91: 172, 92: 173,
            93: 174, 94: 175, 95: 180, 96: 182, 97: 183, 98: 184, 99: 185, 100: 186,
            101: 187, 102: 188, 103: 189, 104: 190
        }

    class_count2 = torch.zeros(225)
    for i, count in enumerate(class_count):
        class_count2[class2mapping.get(i)] = count
        if i==105:
            class_count2[class2mapping.get(i)] = 0
            print("Amount of error =", count)
    class_distrib = class_count2.view(15, 15)  # Assuming 10x10 grid, adjust accordingly

    plt.imshow(class_distrib.numpy())
    plt.colorbar()
    plt.title("Empirical distribution of classes in training data")
    plt.show()

    class_count = torch.where(class_count2 > 0, torch.ones_like(class_count2), class_count2)
    class_used = class_count.view(15, 15)  # Assuming 10x10 grid, adjust accordingly
    plt.imshow(class_used.numpy())
    plt.colorbar()
    plt.title("class_used")
    plt.show()

    return 0

if __name__ == "__main__":
    # get_empirical_distribution(path="../data_test", cl=False)
    main()