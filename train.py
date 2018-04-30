"""
UNet
Train Unet model
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataset import get_train_valid_loader
from model import UNet2
from utils import Option
from dataset import show_pred
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt

def train(model, train_loader, criterion, epoch):
    model.train()
    num_batches = 0
    avg_loss = 0
    with open('logs.txt', 'a') as file:
        for batch_idx, sample_batched in enumerate(train_loader):
            image = sample_batched['image']
            label = sample_batched['label']
            image, label = Variable(image.type(Option.dtype)), Variable(label.type(Option.dtype))
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label/255.)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            num_batches += 1
        avg_loss /= num_batches
        print('epoch: ' + str(epoch) + ', train loss: ' + str(avg_loss))
        file.write('epoch: ' + str(epoch) + ', train loss: ' + str(avg_loss) + '\n')

def val(model, val_loader, criterion, epoch):
    model.eval()
    num_batches = 0
    avg_loss = 0
    with open('logs.txt', 'a') as file:
        for batch_idx, sample_batched in enumerate(val_loader):
            image = sample_batched['image']
            label = sample_batched['label']
            image, label = Variable(image.type(Option.dtype)), Variable(label.type(Option.dtype))
            output = model.forward(image)
            loss = criterion(output, label/255.)
            avg_loss += loss.item()
            num_batches += 1
        avg_loss /= num_batches
        # avg_loss /= len(val_loader.dataset)

        print('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss))
        file.write('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss) + '\n')

# train and validation
def run(model, train_loader, val_loader, criterion):
    for epoch in range(1, Option.epochs):
        train(model, train_loader, criterion, epoch)
        val(model, val_loader, criterion, epoch)

# make prediction
def run_test(model, test_loader):
    """
    predict the masks on testing se
    t
    :param model: trained model
    :param test_loader: testing set
    :param opt: configurations
    :return:
        - predictions: list, for each elements, numpy array (Width, Height)
        - img_ids: list, for each elements, an image id string
    """
    images = []
    predictions = []
    ground_truth = []
    for batch_idx, sample_batched in enumerate(test_loader):
        image = sample_batched['image']
        label = sample_batched['label']
        image = Variable(image.type(Option.dtype))
        output = model.forward(image)
        output = output.data.cpu().numpy()
        output = output.transpose((0, 2, 3, 1))    # transpose to (B,H,W,C)
        label = label.data.cpu().numpy()
        label = label.transpose((0, 2, 3, 1))
        for i in range(0, output.shape[0]):
            pred_mask = np.squeeze(output[i])
            images.append(np.squeeze(image.data.cpu().numpy().transpose((0, 2, 3, 1))[i]))
            predictions.append(pred_mask)
            ground_truth.append(np.squeeze(label[i]))

    return images, predictions, ground_truth

if __name__ == '__main__':
    """Train Unet model"""
    model = UNet2(input_channels=1, nclasses=1)
    if Option.is_train:
        # split all data to train and validation, set split = True
        train_loader, val_loader = get_train_valid_loader(Option.root_dir, batch_size=Option.batch_size,
                                                          shuffle=Option.shuffle,
                                                          num_workers=Option.num_workers,
                                                          pin_memory=Option.pin_memory)
        if Option.n_gpu > 1:
            model = nn.DataParallel(model)
        if Option.is_cuda:
            model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=Option.learning_rate, weight_decay=Option.weight_decay)
        criterion = nn.MSELoss().cuda()
        # start to run a training
        run(model, train_loader, val_loader, criterion)
        # make prediction on validation set
        # predictions, img_ids = run_test(model, val_loader, Option)
        # SAVE model
        if Option.save_model:
            torch.save(model.state_dict(), os.path.join(Option.checkpoint_dir, 'model-emoji.pt'))
    else:
        # load testing data for making predictions
        train_loader, val_loader = get_train_valid_loader(Option.root_dir, batch_size=Option.batch_size,
                                                          shuffle=Option.shuffle,
                                                          num_workers=Option.num_workers,
                                                          pin_memory=Option.pin_memory)
        # load the model and run test
        model.load_state_dict(torch.load(os.path.join(Option.checkpoint_dir, 'model-MRI.pt')))
        if Option.n_gpu > 1:
            model = nn.DataParallel(model)
        if Option.is_cuda:
            model = model.cuda()
        images, predictions, ground_truth = run_test(model, val_loader)
        show_pred(images, predictions, ground_truth)
