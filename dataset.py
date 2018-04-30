"""
UNet
opturations and data loading code for Kaggle Data Science Bowl 2018
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from PIL import Image
from utils import Option
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split


"""Transforms:
Data augmentation
"""
class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if isinstance(self.output_size, int):
            new_h = new_w = self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # resize the image,
        # preserve_range means not normalize the image when resize
        img = transform.resize(image, (new_h, new_w), preserve_range=True, mode='constant')
        label = transform.resize(label, (new_h, new_w), preserve_range=True, mode='constant')
        return {'image': img, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # if sample.keys
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.expand_dims(image, 0)
        label = np.expand_dims(label, 0)
        return {'image': torch.from_numpy(image.astype(np.uint8)),
                'label': torch.from_numpy(label.astype(np.uint8))}

# Helper function to show a batch
def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, masks_batch = sample_batched['image'].numpy().astype(np.uint8), sample_batched['mask'].numpy().astype(np.bool)
    batch_size = len(images_batch)
    for i in range(batch_size):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.tight_layout()
        plt.imshow(images_batch[i].transpose((1, 2, 0)))
        plt.subplot(1, 2, 2)
        plt.tight_layout()
        plt.imshow(np.squeeze(masks_batch[i].transpose((1, 2, 0))))

def show_pred(images, predictions, ground_truth):
    # choose 10 indice from images and visualize them
    indice = [np.random.randint(0, len(images)) for i in range(40)]
    for i in range(0, 40):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.tight_layout()
        plt.title('deformed image')
        plt.imshow(images[indice[i]])
        plt.subplot(1, 3, 2)
        plt.tight_layout()
        plt.title('predicted mask')
        plt.imshow(predictions[indice[i]])
        plt.subplot(1, 3, 3)
        plt.tight_layout()
        plt.title('ground truth label')
        plt.imshow(ground_truth[indice[i]])
    plt.show()

# Load Data Science Bowl 2018 training dataset
class ShapesDataset(Dataset):
    def __init__(self, root_dir, train = True, transform=None):
        if train == True:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')
        self.transform = transform
        self.images_path, self.labels_path = self.get_path()

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        label_path = self.labels_path[idx]
        image = Image.open(image_path).convert('L')
        label = Image.open(label_path).convert('L')
        sample = {'image':np.array(np.abs(image)), 'label':np.array(np.abs(label))}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_path(self):
        images = []
        labels = []
        for root, dirs, files in os.walk(self.root_dir):
            for name in files:
                if name == 'image.png':
                    images.append(os.path.join(root, name))
                else:
                    labels.append(os.path.join(root, name))
        #
#
        return images, labels

def get_train_valid_loader(root_dir, batch_size=16, shuffle=False,
                           num_workers=1, pin_memory=False):

    """Utility function for loading and returning training and validation Dataloader
    :param root_dir: the root directory of data set
    :param batch_size: batch size of training and validation set
    :param split: if split data set to training set and validation set
    :param shuffle: if shuffle the image in training and validation set
    :param num_workers: number of workers loading the data, when using CUDA, set to 1
    :param val_ratio: ratio of validation set size
    :param pin_memory: store data in CPU pin buffer rather than memory. when using CUDA, set to True
    :return:
        if split the data set then returns:
        - train_loader: Dataloader for training
        - valid_loader: Dataloader for validation
        else returns:
        - dataloader: Dataloader of all the data set
    """
    train_transformed_dataset = ShapesDataset(root_dir=root_dir,
                                              train = True,
                                               transform=transforms.Compose([
                                                  Rescale(Option.img_width),
                                                  ToTensor()
                                               ]))
    val_transformed_dataset = ShapesDataset(root_dir=root_dir,
                                            train=False,
                                             transform=transforms.Compose([
                                                  Rescale(Option.img_width),
                                                  ToTensor()
                                               ]))


    train_loader = DataLoader(train_transformed_dataset,batch_size=batch_size,
                              shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_transformed_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return (train_loader, val_loader)


if __name__ == '__main__':

    # shape = ShapesDataset(Option.root_dir,
    #                       True,
    #                       transform=transforms.Compose([
    #                                               Rescale(Option.img_width),
    #                                               ToTensor()
    #                                            ]))
    # print(shape[0])

    train_loader, val_loader = get_train_valid_loader(Option.root_dir, batch_size=Option.batch_size,
                                                        shuffle=Option.shuffle,
                                                        num_workers=Option.num_workers,
                                                        pin_memory=Option.pin_memory)

    for i_batch, sample_batched in enumerate(val_loader):
        print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())
        for i in range(0, Option.batch_size):
            image = sample_batched['image'][i].numpy()
            label = sample_batched['label'][i].numpy()
            fig = plt.figure()
            plt.subplot(1, 2, 1)
            plt.title('Deformed image')
            plt.imshow(np.squeeze(image))
            plt.subplot(1, 2, 2)
            plt.title('Ground truth label')
            plt.imshow(np.squeeze(label))
            # fig.savefig('./images/deformed_'+ str(i_batch) + '_' + str(i), bbox_inches='tight', dpi=150)
        # plt.show()
        # break