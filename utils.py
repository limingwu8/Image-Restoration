"""
Image Registration
Common utility functions and classes
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from skimage import io
from PIL import Image
import numpy as np
import torch
from os.path import basename
from skimage.morphology import label
import pandas as pd
import matplotlib.pylab as plt
from deforme import deforme
import cv2


# Base Configuration class
# Don't use this class directly. Instead, sub-class it and override

class Config(object):

    name = None
    img_width = 256
    img_height = 256
    img_channel = 3
    batch_size = 16
    learning_rate = 1e-3
    learning_momentum = 0.9
    weight_decay = 1e-4
    shuffle = False

    def __init__(self):
        self.IMAGE_SHAPE = np.array([
            self.img_width, self.img_height, self.img_channel
        ])

    def display(self):
        """Display Configuration values"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

# Configurations

class Option(Config):
    """Configuration for training on Kaggle Data Science Bowl 2018
    Derived from the base Config class and overrides specific values
    """
    name = "Shapes"

    # root dir of deformed dataset
    root_dir = '/home/liming/Documents/dataset/dtidata'

    img_width = 128
    img_height = 128

    num_workers = 4     	# number of threads for data loading
    shuffle = True      	# shuffle the data set
    batch_size = 16     	# GTX1060 3G Memory
    epochs = 50			    # number of epochs to train
    is_train = False     	# True for training, False for making prediction
    save_model = False   	# True for saving the model, False for not saving the model
    n_gpu = 1				# number of GPUs
    learning_rate = 1e-3    # learning rate
    weight_decay = 1e-5		# weight decay
    pin_memory = True   	# use pinned (page-locked) memory. when using CUDA, set to True
    is_cuda = torch.cuda.is_available()  	# True --> GPU
    num_gpus = torch.cuda.device_count()  	# number of GPUs
    checkpoint_dir = "./checkpoint"  		# dir to save checkpoint
    dtype = torch.cuda.FloatTensor if is_cuda else torch.Tensor  # data type

"""
Create deformed images:
Read images and deforming them and save them to another folder
"""
class Shapes(object):
    def __init__(self, src_dir, dest_dir):
        self.opt = Option
        self.src_dir = src_dir
        self.dest_dir = dest_dir

    # read all images, deforming them and save to other folder
    def deforme(self):
        print('generating deformed images starts...')
        for root, dirs, files in os.walk(self.src_dir):
            root_dir_name = basename(root)
            for name in files:
                img_id = name.split('.')[0]
                path = os.path.join(self.dest_dir, root_dir_name, img_id)
                if not os.path.exists(path):
                    os.makedirs(path)
                label = Image.open(os.path.join(root, name)).convert('L')
                # save deformed image
                image = deforme(np.array(label))
                image = Image.fromarray(image)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(os.path.join(path, 'image.png'))
                # save label
                label.save(os.path.join(path, 'label.png'))
        print('generating deformed images done...')

"""
Create deformed images:
Read emoji images and deforming them and save them to another folder
"""
class Emoji(object):
    def __init__(self, src_dir, dest_dir):
        self.opt = Option
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.img_size = 225
        self.save_img = 1

    def deforme_train(self):
        # generate preprocessed image for training
        print('generating deformed images starts...')
        for root, dirs, files in os.walk(self.src_dir):
            root_dir_name = basename(root)
            i = 0
            for name in files:
                print('processing ',name)
                # for each emoji, generate 10 deformed images
                try:
                    for j in range(50):
                        path = os.path.join(self.dest_dir, str(i))
                        if not os.path.exists(path):
                            os.makedirs(path)
                        object = Image.open(os.path.join(root, name), 'r')
                        object_size = object.size[0]
                        img = Image.new('L', (self.img_size, self.img_size), 0)
                        img.paste(object, (int((self.img_size - object_size) / 2), int((self.img_size - object_size) / 2)), mask=object)
                        label = img
                        # save deformed image
                        image = deforme(np.array(label))
                        image = self.noisy(image)
                        plt.imshow(image)
                        image = Image.fromarray(image)
                        if image.mode != 'L':
                            image = image.convert('L')
                        if self.save_img:
                            image.save(os.path.join(path, 'image.png'))
                            # save label
                            label.save(os.path.join(path, 'label.png'))
                        i += 1
                except:
                    print('Something wrong!')
        print('generating deformed images done...')

    def deforme_val(self):
        # generate preprocessed image for validation
        print('generating deformed images starts...')
        for root, dirs, files in os.walk(self.src_dir):
            root_dir_name = basename(root)
            i = 0
            for name in files:
                print('processing ', name)
                # for each emoji, generate 10 deformed images
                try:
                    for j in range(1):
                        path = os.path.join(self.dest_dir, str(i))
                        if not os.path.exists(path):
                            os.makedirs(path)
                        object = Image.open(os.path.join(root, name), 'r')
                        object_size = object.size[0]
                        img = Image.new('L', (self.img_size, self.img_size), 0)
                        img.paste(object, (int((self.img_size - object_size) / 2), int((self.img_size - object_size) / 2)), mask=object)
                        label = img
                        # save deformed image
                        image = deforme(np.array(label))
                        image = self.noisy(image)
                        image = Image.fromarray(image)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        if self.save_img:
                            image.save(os.path.join(path, 'image.png'))
                            # save label
                            label.save(os.path.join(path, 'label.png'))
                        i += 1
                except:
                    print('Something wrong!')
        print('generating deformed images done...')

    def noisy(self, image):
        # add Gaussian noise to an image
        row, col = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = 10*np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        gauss = np.abs(gauss)
        # noisy = np.clip(image + gauss, 0, 255)
        noisy = image + gauss
        return noisy

if __name__ == '__main__':
    """ Prepare deformed dataset
    Read shapes images from src_dir, deforming them and save them to dest_dir
    """
    # src_dir = '/home/liming/Documents/dataset/shapes/four-shapes/shapes'
    # dest_dir = '/home/liming/Documents/dataset/shapes/four-shapes/deformed-shapes'
    #
    # shapes = Shapes(src_dir, dest_dir)
    # shapes.deforme()

    """ Prepare deformed dataset for training
    Read emoji images from src_dir, deforming them and save them to dest_dir
    """
    src_dir = '/home/liming/Documents/dataset/dtidata/raw_train'
    dest_dir = '/home/liming/Documents/dataset/dtidata/train'

    emoji = Emoji(src_dir, dest_dir)
    emoji.deforme_train()

    """ Prepare deformed dataset for validation
    Read emoji images from src_dir, deforming them and save them to dest_dir
    """
    # src_dir = '/home/liming/Documents/dataset/dtidata/raw_val'
    # dest_dir = '/home/liming/Documents/dataset/dtidata/val'
    #
    # emoji = Emoji(src_dir, dest_dir)
    # emoji.deforme_val()