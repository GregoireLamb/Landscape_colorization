import numpy as np
import progress
import torch
import time
import os
import sys
from alive_progress import alive_bar

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Cu_net import *
from src.GrayscaleImageFolder import *
from src.util import *
import click

def main():
    click.clear()

    # Check if GPU is available
    use_gpu = torch.cuda.is_available()
    model = Cu_net()
    epochs = 1
    batch_size = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

    print("[LANDSCAPE COLORIZATION]\n")
    print("Parameters:")
    print("\tModel: {}".format(model.name))
    print("\tUsing GPU: {}".format(use_gpu))
    print("\tModel: {}".format(model.name))
    print("\tEpochs: {}".format(epochs))
    print("\tBatch size: {}".format(batch_size))
    print("\tCriterion: {}".format(criterion))
    print("\tOptimizer: {}\n".format(optimizer.__class__.__name__))


    train_transforms = transforms.Compose([])
    train_imagefolder = GrayscaleImageFolder('../data_train', train_transforms)
    train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=batch_size, shuffle=True)

    val_transforms = transforms.Compose([])
    val_imagefolder = GrayscaleImageFolder('../data_test', val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=batch_size, shuffle=False)

    if use_gpu:
        criterion = criterion.cuda()
        model = model.cuda()

    # Make folders and set parameters
    os.makedirs('outputs/color', exist_ok=True)
    os.makedirs('outputs/gray', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    save_images = True
    best_losses = 1e10

    # Train model
    print("Start training")
    for epoch in range(epochs):
        # Train for one epoch, then validate
        train(train_loader, model, criterion, optimizer, epoch)
        with torch.no_grad():
            losses = validate(val_loader, model, criterion, save_images, epoch)
        # Save checkpoint and replace old best model if current model is better
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
  color_image = lab2rgb(color_image.astype(np.float64))

  grayscale_input = grayscale_input.squeeze().numpy()
  if save_path is not None and save_name is not None:
    plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
    plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))


def validate(val_loader, model, criterion, save_images, epoch,use_gpu = True):
    model.eval()

    # Prepare value counters and timers
    # batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    # end = time.time()
    losses = 0
    count = 0
    already_saved_images = False
    with alive_bar(total=len(val_loader), title="Valid epoch: [{0}]".format(epoch),
                   spinner='classic') as bar:  # len(train_loader) = n_batches
        for i, (input_gray, input_ab, target) in enumerate(val_loader):
            # data_time.update(time.time() - end)
            if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

            # Run model and record loss
            output_prob = model(input_gray) # -> batch*256*256*100
            output_prob = torch.flatten(output_prob, start_dim=2).cuda()
            input_ab_class = torch.flatten(ab2class(input_ab), start_dim=1).cuda()
            loss = criterion(output_prob, input_ab_class)
            losses += loss.item()
            count += 1

            unflatten = torch.nn.Unflatten(2, (256, 256))# TODO adapt hard coded values
            output_prob = unflatten(output_prob)
            output_ab = prob2class(output_prob)
            output_ab = class2ab(output_ab)
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


def train(train_loader, model, criterion, optimizer, epoch, use_gpu = True, save_images = True):
    model.train()

    with alive_bar(total=len(train_loader), title="Train epoch: [{0}]".format(epoch), spinner='classic') as bar: #len(train_loader) = n_batches
        for i, (input_gray, input_ab, target) in enumerate(train_loader):
            if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

            output_ab_class = model(input_gray)
            input_ab_class= ab2class(input_ab)
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


def lab_to_rgb(l, ab):
    plt.clf()  # clear matplotlib
    color_image = torch.cat((l, ab), 0).numpy()  # combine channels
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    color_image = lab2rgb(color_image.astype(np.float64))
    #print image
    plt.imshow(color_image)
    plt.show()

if __name__ == "__main__":
    main()
    # test()