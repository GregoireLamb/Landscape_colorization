import torch

from src.Color_cnn import *
from src.Cu_net import *
from src.GrayscaleImageFolder import *
from src.util import *
import cv2

def main():
    # Check if GPU is available
    use_gpu = torch.cuda.is_available()
    print("Using GPU: {}".format(use_gpu))
    # model = Color_cnn()
    model = Cu_net()

    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)
    batch_size = 8

    train_transforms = transforms.Compose([])
    train_imagefolder = GrayscaleImageFolder('../data_train', train_transforms)
    train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=batch_size, shuffle=True)

    val_transforms = transforms.Compose([])
    val_imagefolder = GrayscaleImageFolder('../data_test', val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=batch_size, shuffle=False)

    # Move model and loss function to GPU
    if use_gpu:
        print("Using GPU")
        criterion = criterion.cuda()
        model = model.cuda()

    # Make folders and set parameters
    os.makedirs('outputs/color', exist_ok=True)
    os.makedirs('outputs/gray', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    save_images = True
    best_losses = 1e10
    epochs = 20

    # Train model
    for epoch in range(epochs):
        # Train for one epoch, then validate
        train(train_loader, model, criterion, optimizer, epoch)
        with torch.no_grad():
            print("Validating")
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


def validate(val_loader, model, criterion, save_images, epoch):
  model.eval()

  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  already_saved_images = False
  for i, (input_gray, input_ab, target) in enumerate(val_loader):
    data_time.update(time.time() - end)

    # Use GPU
    use_gpu = True
    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    # Run model and record loss
    output_prob = model(input_gray) # -> batch*256*256*100
    output_prob = torch.flatten(output_prob, start_dim=2).cuda()
    input_ab_class = torch.flatten(ab2class(input_ab), start_dim=1).cuda()
    loss = criterion(output_prob, input_ab_class)
    losses.update(loss.item(), input_gray.size(0))

    print("output_ab_class: ", output_prob.shape)
    #TODO change hard coded values
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
        # turn 100  -> to  1
        to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

    # Record time to do forward passes and save images
    batch_time.update(time.time() - end)
    end = time.time()

    # Print model accuracy -- in the code below, val refers to both value and validation
    if i % 25 == 0:
      print('Validate: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
             i, len(val_loader), batch_time=batch_time, loss=losses))

  print('Finished validation.')
  return losses.avg


def train(train_loader, model, criterion, optimizer, epoch):
    use_gpu = True
    print('Starting training epoch {}'.format(epoch))
    model.train()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, (input_gray, input_ab, target) in enumerate(train_loader):
        # Use GPU if available
        if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

        # Record time to load data (above)
        data_time.update(time.time() - end)
        # Run forward pass
        output_ab_class = model(input_gray)
        input_ab_class= ab2class(input_ab)
        if use_gpu: output_ab_class, input_ab_class, target = output_ab_class.cuda(), input_ab_class.cuda(), target.cuda()

        output_ab_class = torch.flatten(output_ab_class, start_dim=2)
        print("dim output_ab_class: ", output_ab_class.shape)
        input_ab_class = torch.flatten(input_ab_class, start_dim=1).long()
        print("dim input_ab_class: ", input_ab_class.shape)

        # desire shape is batch, Q, x for output and batch, x for input
        loss = criterion(output_ab_class,input_ab_class)
        losses.update(loss.item(), input_gray.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 25 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    print('Finished training epoch {}'.format(epoch))


def test():
    batch_size = 2
    test_transforms = transforms.Compose([])
    test_imagefolder = GrayscaleImageFolder('../data_test', test_transforms)
    test_loader = torch.utils.data.DataLoader(test_imagefolder, batch_size=batch_size, shuffle=False)

    # Move model and loss function to GPU
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using GPU")
        # model = model.cuda()

    # model.eval()

    for i, (input_gray, input_ab, target) in enumerate(test_loader):
        use_gpu = True
        if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

        # output_ab = model(input_gray)

        for z in range(len(input_ab)):
            lab_to_rgb(input_gray[z].cpu(), input_ab[z].detach().cpu())

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

def diff_imgs(imgA, imgB):
    #load images
    imgA = cv2.imread(imgA)
    imgB = cv2.imread(imgB)
    # return true if they are the same
    return np.array_equal(imgA, imgB)

if __name__ == "__main__":
    # load one image from data_test
    main()
    # test()
    # print(diff_imgs('outputs/color/img-0-epoch-0.jpg', 'outputs/color/img-0-epoch-9.jpg'))

    #creata a torch tensor of dim 1 2 128 128 with only zeros
    # a = torch.zeros(1, 2, 128, 128)
    # a[0,1,125,125] = 1