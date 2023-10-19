from src.Color_cnn import *
from src.GrayscaleImageFolder import *
from src.util import *


def main():
    # Check if GPU is available
    use_gpu = torch.cuda.is_available()
    print("Using GPU: {}".format(use_gpu))
    model = Color_cnn()

    """Def criterion and optimizer"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

    # Training
    #train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()])
    # train_imagefolder = GrayscaleImageFolder('../data_train', train_transforms)

    train_transforms = transforms.Compose([])
    train_imagefolder = GrayscaleImageFolder('../data_train', train_transforms)
    train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=64, shuffle=True)

    # Validation
    # val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    # val_imagefolder = GrayscaleImageFolder('../data_test', val_transforms)

    val_transforms = transforms.Compose([])
    val_imagefolder = GrayscaleImageFolder('../data_test', val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)

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
    epochs = 2

    # Train model
    for epoch in range(epochs):
        # Train for one epoch, then validate
        train(train_loader, model, criterion, optimizer, epoch)
        with torch.no_grad():
            print("Validating, what happen here")
            losses = validate(val_loader, model, criterion, save_images, epoch)
        # Save checkpoint and replace old best model if current model is better
        if losses < best_losses:
            best_losses = losses
            torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch + 1, losses))

    # Show images
    # import matplotlib.image as mpimg
    # image_pairs = [('outputs/color/img-0-epoch-0.jpg', 'outputs/gray/img-0-epoch-0.jpg'),
    #                ('outputs/color/img-1-epoch-0.jpg', 'outputs/gray/img-1-epoch-0.jpg')]
    # for c, g in image_pairs:
    #     color = mpimg.imread(c)
    #     gray = mpimg.imread(g)
    #     f, axarr = plt.subplots(1, 2)
    #     f.set_size_inches(15, 15)
    #     axarr[0].imshow(gray, cmap='gray')
    #     axarr[1].imshow(color)
    #     axarr[0].axis('off'), axarr[1].axis('off')
    #     plt.show()

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
    output_ab = model(input_gray) # throw away class predictions
    loss = criterion(output_ab, input_ab)
    losses.update(loss.item(), input_gray.size(0))

    # Save images to file
    if save_images and not already_saved_images:
      already_saved_images = True
      for j in range(min(len(output_ab), 10)): # save at most 5 images
        save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
        save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
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
        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
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


if __name__ == "__main__":
    main()
