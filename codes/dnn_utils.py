'''
Created this file for all general dnn utils
'''

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch import nn

def update_lr(learning_rate0, epoch_num, decay_rate):
    """
    Calculates updated the learning rate using exponential weight decay.
    
    Arguments:
    learning_rate0 -- Original learning rate. Scalar
    epoch_num -- Epoch number. Integer
    decay_rate -- Decay rate. Scalar

    Returns:
    learning_rate -- Updated learning rate. Scalar 
    """
    
    learning_rate = (1./(1 + decay_rate * epoch_num)) * learning_rate0
    
    return learning_rate


def schedule_lr_decay(optimizer, learning_rate0, epoch_num, decay_rate=1, time_interval=1000):
    """
    Calculates updated the learning rate using exponential weight decay.
    
    Arguments:
    learning_rate -- Original learning rate. Scalar
    epoch_num -- Epoch number. Integer.
    decay_rate -- Decay rate. Scalar.
    time_interval -- Number of epochs where you update the learning rate.

    Returns:
    learning_rate -- Updated learning rate. Scalar 
    """
    
    if epoch_num % time_interval == 0:
        learning_rate = (1./(1 + decay_rate * (epoch_num / time_interval)))*learning_rate0
    else:
        learning_rate = optimizer.param_groups[0]["lr"]
    
    return learning_rate

def train(epoch,model, train_loader, criterion, device):
    model.train()
#     learn_rate = schedule_lr_decay(optimizer, learning_rate, epoch, decay_rate=1, time_interval = 15)
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = learn_rate
#         cur_lr = optimizer.param_groups[0]["lr"] 
    print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]}')
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(model.state_dict(), './models and saliencies/cifar_model_vgg_'+str(model_id)+'.pth')
            torch.save(optimizer.state_dict(), './models and saliencies/cifar_optimizer_vgg_'+str(model_id)+'.pth')

test_losses = []
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    return test_loss
