import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch import nn
import os
from scipy.stats import truncnorm

from models import ConvModel, VGGNet, LeNet, MNNet, SanityCheckCNN, SanityCheckMLP, FashionCustomizedCNN

import numpy as np
import pandas as pd
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from source.utils import get_argparser
from myscripts.generateSynData import artificial_batching_patterned_space1, artificial_batching_patterned_space2, three_class, actual_spatial, multi_way_data, artificial_batching_trend, my_multi_block_var_data, my_uniform_top_down, generate_top_down_pos_neg, generate_top_down_neg_neg, generate_top_down_zero
from myscripts.stitchWindows import stitch_windows
import time
import sys
import copy

from myscripts.metricMeasureNew import calcMetric, newMetric
from interpretability import post_process_saliency
from PIL import Image
from torchvision import transforms
from torchvision import datasets, models, transforms

import os
import glob

import pickle
import skimage.filters
import cv2
import tqdm
import joblib

device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")


def load_bgc_imagenet_saliency_data():
    
    '''
    Load images whose background have been changed.
    '''
    
    print("Loading imagenet data with background replacement...")
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    data = []
    labels = []

    indices = [2, 19, 20, 40,  55, 64, 83, 105, 250, 256] 

    basename = './bgc_n_edge_detection/ILSVRC2012_cbg_cb_'
    for idx in range(len(indices)):
        fname = basename + '00000' + str(indices[idx]).zfill(3)+ ".JPEG"
        print(fname)
        input_image = Image.open(fname)
        input_tensor = preprocess(input_image)
        data.append(input_tensor)
        labels.append(torch.as_tensor(13))

    data = torch.stack(data, dim=0)
    labels = torch.stack(labels, dim=0)
    labels = labels.long()
    print(data.shape, labels.shape)

    dataLoaderSal = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, labels), batch_size = 1, shuffle=False)
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
        
    return data, labels, dataLoaderSal, categories


def load_imagenet_saliency_data():
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'test': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data = []
    labels = []

    indices = [2, 4, 19, 20, 27, 40, 55, 64, 83, 94, 105, 145, 169, 172, 242, 250, 256, 259, 260, 263]

    basename = './ImageNet/test/ILSVRC2012_test_'
    for idx in range(len(indices)):
        fname = basename + '00000' + str(indices[idx]).zfill(3)+ ".JPEG"
        print(fname)
        input_image = Image.open(fname)
        input_tensor = data_transforms['val'](input_image)
        data.append(input_tensor)
        labels.append(torch.as_tensor(13))

    data = torch.stack(data, dim=0)
    labels = torch.stack(labels, dim=0)
    labels = labels.long()
    print(data.shape, labels.shape)

    dataLoaderSal = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, labels), batch_size = 1, shuffle=False)
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
        
    return data, labels, dataLoaderSal, categories

def load_imgnet_val_data():

    with open('./imgnet_val_data/val224.pkl', 'rb') as f:
        d = joblib.load(f)

    # Read the imageNet categories
    with open("./imgnet_val_data/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    images = d['data'].transpose(0, 2, 3, 1)
    labels = d['target']
    print(images.shape)
    f.close()
    
    data_transforms = {
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    unnormalized_resized_images = []
    normalized_images = []
    targets = []

    image_idx = 0
    while image_idx < 200:
        
        input_image = images[image_idx]
        
        input_tensor = data_transforms['val'](input_image)
        normalized_images.append(input_tensor)
        targets.append(torch.as_tensor(labels[image_idx]))
        unnormalized_resized_images.append(input_image)
        image_idx += 1

    normalized_images = torch.stack(normalized_images, dim=0)
    targets = torch.stack(targets, dim=0)
    targets = targets.long()

    dataLoaderSal = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(normalized_images, targets), batch_size = 1, shuffle=False)
    

    return unnormalized_resized_images, normalized_images, targets, dataLoaderSal, categories

def load_imagenet_saliency_metric_eval_data():
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'test': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    images = []
    normalized_images = []
    labels = []

    indices = [2, 4, 19, 20, 27, 40, 55, 64, 83, 94, 105, 145, 169, 172, 242, 250, 256, 259, 260, 263]

    basename = './ImageNet/test/ILSVRC2012_test_'
    idx = 1
    image_count = 0
    while image_count < 200:
        fname = basename + '00000' + str(idx).zfill(3)+ ".JPEG"
#         print(fname)
        input_image = Image.open(fname)
        idx += 1
        
        if len(np.array(input_image).shape) < 3:
            continue
        input_tensor = data_transforms['val'](input_image)
        normalized_images.append(input_tensor)
        labels.append(torch.as_tensor(13))
        images.append(input_image)
        image_count += 1

    normalized_images = torch.stack(normalized_images, dim=0)
    labels = torch.stack(labels, dim=0)
    labels = labels.long()

    dataLoaderSal = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(normalized_images, labels), batch_size = 1, shuffle=False)
    
    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
        
    return images, normalized_images, labels, dataLoaderSal, categories


def load_cifar10_saliency_data():
    normalize_original = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    transform_train = transforms.Compose([transforms.Resize(256),  #resises the image so it can be perfect for our model.
                                      transforms.CenterCrop(224),
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #zooms, change shear angles.
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                      normalize, #Normalize all the images
                               ])
 
    transform = transforms.Compose([transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           normalize
                           ])
    
    training_dataset = datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform_train) 
    test_dataset = datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True) 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 100, shuffle=False)

    saliency_segment = list(range(200))
    saliency_dataset = torch.utils.data.Subset(test_dataset, saliency_segment)

    dataLoaderSal = torch.utils.data.DataLoader(saliency_dataset, batch_size=1,
                                        shuffle=False, num_workers=2)
    
    dataLoaderFullSal = torch.utils.data.DataLoader(saliency_dataset, batch_size=200,
                                        shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    dataiter = iter(dataLoaderFullSal) # converting our train_dataloader to iterable so that we can iter through it. 
    images, labels = dataiter.next() #going from 1st batch of 100 images to the next batch
    
    return images, labels, dataLoaderSal, classes


def load_cifar10_original_saliency_data():
    print('Hello!')
    normalize_original = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    
    transform_train = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #zooms, change shear angles.
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                      normalize_original, #Normalize all the images
                               ])
 
    transform = transforms.Compose([transforms.Resize((32,32)),
                           transforms.ToTensor(),
                           normalize_original
                           ])
    
    training_dataset = datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform_train) 
    test_dataset = datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True) 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 100, shuffle=False)

    saliency_segment = list(range(200))
    saliency_dataset = torch.utils.data.Subset(test_dataset, saliency_segment)

    dataLoaderSal = torch.utils.data.DataLoader(saliency_dataset, batch_size=1,
                                        shuffle=False, num_workers=2)
    
    dataLoaderFullSal = torch.utils.data.DataLoader(saliency_dataset, batch_size=200,
                                        shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    dataiter = iter(dataLoaderFullSal) # converting our train_dataloader to iterable so that we can iter through it. 
    images, labels = dataiter.next() #going from 1st batch of 100 images to the next batch
    
    return images, labels, dataLoaderSal, classes


def load_mnist_saliency_data():
    
    print('Loading MNIST data')
    trainX, trainY = torch.load('./my_mnist/MNIST/processed/training.pt')
    testX, testY = torch.load('./my_mnist/MNIST/processed/test.pt')
    trainX = trainX.unsqueeze(1).float()
    trainX /= 255
    trainY = trainY.long()
    testX = testX.unsqueeze(1).float()
    testX /= 255
    testY = testY.long()

    transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])

    trainX = transform(trainX)
    testX = transform(testX)
    
    dataLoaderSal = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(testX[9000:], testY[9000:]), batch_size = 1, shuffle=False)
    
    classes = ('zero', 'one', 'two', 'three',
           'four', 'five', 'six', 'seven', 'eight', 'nine')
    
    return testX[9000:], testY[9000:], dataLoaderSal, classes



def load_fmnist_saliency_data():
    
    print("Loading fashion MNIST data.")
    trainX, trainY = torch.load('./fashion_mnist/FashionMNIST/processed/training.pt')
    testX, testY = torch.load('./fashion_mnist/FashionMNIST/processed/test.pt')
    trainX = trainX.unsqueeze(1).float()
    trainX /= 255
    trainY = trainY.long()
    testX = testX.unsqueeze(1).float()
    testX /= 255
    testY = testY.long()

    batch_size_train = 64
    batch_size_test = 1000

    transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Normalize(
                                     (0.2860,), (0.3530,))
                                 ])

    trainX = transform(trainX)
    testX = transform(testX)
    
    dataLoaderSal = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(testX[9500:] , testY[9500:]), 
            batch_size=1, shuffle=False)

    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
#     trainX = torch.clamp(trainX, -1, 1)
#     trainX = (trainX + 1)/2.0
#     testX = torch.clamp(testX, -1, 1)
#     testX = (testX + 1) / 2.0

#     transform=torchvision.transforms.Compose([transforms.Resize((224,224)),
#                                torchvision.transforms.Normalize(
#                                  (0.2860,), (0.3530,))
#                              ])

#     trainX = transform(trainX)
#     testX = transform(testX)

#     trainX = trainX.repeat(1, 3, 1, 1)
#     testX = testX.repeat(1, 3, 1, 1)

#     trainX = torch.clamp(trainX, -1, 1)
#     trainX = (trainX + 1)/2.0
#     testX = torch.clamp(testX, -1, 1)
#     testX = (testX + 1) / 2.0
    
    return testX[9500:], testY[9500:], dataLoaderSal, classes


def ReLU(x):
    return x * (x > 0)
    
def plot_processed_maps(FinalData, Labels, saliency_dict, data_name, method_captions, p):
    
    
    class_labels = {'cifar10': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
                   'mnist': ('zero', 'one', 'two', 'three',
           'four', 'five', 'six', 'seven', 'eight', 'nine')}
    
    classnames = class_labels[data_name]

    filenamepath = "./models and saliencies"
    filenamepath = os.path.join(filenamepath, 'saliency')
    
    id = len(saliency_dict)
    print(id)
    
    D = np.asarray(range(395, 405, 1))  # Using: MNIST 395 - 405, fMNIST: 10 - 20
    print(D)
    L = Labels[D[:]]
    print('Labels:', L)
    
    if data_name == 'cifar10':
        
        model = VGGNet(3, 10) # CIFAR model
        path = './models and saliencies/method_research_cifar_model_vgg_customized_new_'+str(p)+'.pth'
        
    elif data_name == 'mnist':
        model = SanityCheckCNN()
        path = './models and saliencies/method_research_mnist_model_'+str(p)+'.pth'
    
    model_dict = torch.load(path, map_location=device)  # with good components
    model.load_state_dict(model_dict)
    model.to(device)
    print('Model loaded from:', path)
    
    model.eval()
    with torch.no_grad():
        output = model(FinalData) # We don't need to unsqueeze(1) if we already have that dimension as in color image
    
    preds = output.data.max(1, keepdim=True)[1]
    print(preds.shape)
    print(torch.flatten(preds[D[:]]))


    fig, axes = plt.subplots(10, id+1, figsize=(18,18))

    for i in range(10):
        sample = FinalData[D[i]].numpy()
        print("Inside Plots:{}".format(sample.shape))
        sample = np.clip(sample, -1, 1)
        sample = (sample + 1.0)/2.0
        
        if np.squeeze(sample).ndim == 3:
            sample = sample.transpose(1,2,0)
            
        h= axes[i, 0].imshow(np.squeeze(sample), interpolation=None, aspect='auto', cmap=plt.cm.inferno) 
        lim = np.amax(np.squeeze(sample))

        fig.colorbar(h, ax=axes[i,0])
        h.set_clim(0, lim)

        axes[i, 0].axes.xaxis.set_ticks([])
        axes[i, 0].axes.yaxis.set_ticks([])

        axes[i,0].set_ylabel(str(int(Labels[D[i]])), fontsize=12)

    for method_id in range(id):

        saliency = saliency_dict[method_id]

        for i in range(10):

            sample = saliency_dict[method_id][D[i]]
            
            if np.squeeze(sample).ndim == 3:
                sample = np.sum(np.abs(sample), axis=0)

            sample = sample / np.max(sample)

            axes[i, method_id + 1].imshow(sample, interpolation=None, aspect='auto', cmap='Reds')
            axes[i, method_id + 1].set_title(str(classnames[preds[D[i]].item()]),fontsize='small' )

    #     Turn off *all* ticks & spines, not just the ones with colormaps.

            axes[i, method_id + 1].axes.xaxis.set_ticks([])
            axes[i, method_id + 1].axes.yaxis.set_ticks([])

    axes[0, 0].set_title('Data', fontsize='medium')


    for method_id in range(id):
        axes[0, method_id + 1].set_title(method_captions[method_id], fontsize='medium')

    plt.show()

    path = os.path.join('./Plots/Real/', "method_research_"+data_name+'_iclr_plots_preAbsolute_'+str(p) )
    print(path)
    fig.savefig(path+'.svg', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
    fig.savefig(path+'.pdf', format='pdf', dpi=300)
    fig.savefig(path+'.png', format='png', dpi=300)
    print('Plots Saved...', path)
    plt.close(fig)

    
    
def generate_post_processed_maps(data, fname, method_list=None, random_seeds=None, viz=True, scale_flag=False):
    
    
    Dataset = {0: 'mnist', 1: 'fmnist', 2: 'cifar10', 3: 'imgnet'}
    
    DatasetDict = {"mnist": load_mnist_saliency_data, "fmnist": load_fmnist_saliency_data,
                   'cifar10': load_cifar10_saliency_data, 'imgnet': load_imagenet_saliency_data}
    
    FinalData, Labels, dataLoaderSal, classes = DatasetDict[Dataset[data]]()
    print('Final Data Shape:', FinalData.shape)

    print('Test Data Shape:', Labels.shape)

    
    prefix_gd = fname+'_Grad_zero_'
    prefix_gd_input = fname+'_InputXGrad_zero_'
    prefix_ig = fname+'_IG_zero_'
    prefix_capt_ig = fname+'_CaptumIG_zero_'
    prefix_local_ig= fname+'_LocalIGAll_zero_'
    
    prefix_only_ig= fname+'_LocalIGAll_zero_only_ig_'
    prefix_only_max= fname+'_LocalIGAll_zero_only_max_'
    prefix_path_grad= fname+'_LocalIGAll_zero_path_grad_'
    
    
    prefix_sgig = fname+'_SGIG_zero_'
    prefix_sgsqig = fname+'_SGSQIG_zero_'
    prefix_vgig = fname+'_VGIG_zero_'
    
    prefix_gradasc =fname+'_GradAsc_iter_200_lr_0.0001_zero_'
    prefix_multigradasc = fname+'_MultiGradAsc_iter_200_lr_0.0001_zero_'
    prefix_wt = fname+'_WtPerturbation_iter_200_zero_'
    
    prefix_gbp = fname+'_GBP_zero_'
    prefix_dlift= fname+'_DLift_zero_'
    prefix_lrp = fname+'_LRP_zero_'
    prefix_guidgradCam= fname+'_GuiGradCAM_zero_'


    all_methods = {"GD": prefix_gd, "Inp X GD": prefix_gd_input, "IG": prefix_ig, "CaptIG": prefix_capt_ig, \
                  "L.IG": prefix_local_ig, "ONLY.IG": prefix_only_ig, "ONLY.M": prefix_only_max, "SGIG": prefix_sgig, \
                  "SGSQIG": prefix_sgsqig, "VGIG": prefix_vgig, "GDAsc": prefix_gradasc, "M.GDAsc": prefix_multigradasc, \
                   "Wt.P": prefix_wt, "GBP": prefix_gbp, "DL": prefix_dlift, "LRP": prefix_lrp, "GradCAM": prefix_guidgradCam}

    filenamepath = "./models and saliencies"
    filenamepath = os.path.join(filenamepath, 'saliency')
    
    filenames, titles = [], []
    
    for method in method_list:
        filenames.append(all_methods[method])
        print(filenames[-1])
        
        if method not in ["GDAsc", "M.GDAsc", "Wt.P"]:
            titles.extend([method])
        else:
            titles.extend([method+'.IG', method+'.M'])
        
    print("Titles for Chosen Methods: ", titles)

    post_processed_maps = []
    saliency_dict = {}

    for p in random_seeds:

        id = 0

        print("Processing of gradients files started...")
        for fname in filenames:
            fname = os.path.join(filenamepath, fname+str(p)+'.npy')
            print(fname)
            all_gradients = np.load(fname)
           

            all_gradients = np.nan_to_num(all_gradients)
#             all_gradients = np.abs(all_gradients)

            print(all_gradients.shape)


            array_has_nan = np.isnan(all_gradients).any()
            input_has_nan = np.isnan(FinalData).any()
            print('Grads have NaNs:', array_has_nan)
            print('Inputs have NaNs:', input_has_nan)


            if "Grad" in fname and not 'XGrad' in fname and not "CAM" in fname and \
                len(all_gradients.shape) == FinalData.squeeze().dim():
                
                saliency_dict[id] = all_gradients
                id += 1
                print(f'Current ID: {id}')
                
            elif len(all_gradients.shape) == FinalData.squeeze().dim():
                saliency_dict[id] = all_gradients
                id += 1
                print(f'Current ID: {id}')

            elif "MultiDir" in fname and len(all_gradients.shape) == FinalData.squeeze().dim() + 1:
                max_sal, ig_like_sal, avg_sal = post_process_saliency(FinalData, all_gradients, scaling=scale_flag)
                saliency_dict[id] = max_sal
                saliency_dict[id + 1] = ig_like_sal
                saliency_dict[id + 2] = avg_sal
                id += 3
                print(f'Current ID: {id}')
            elif len(all_gradients.shape) == FinalData.squeeze().dim() + 1:
                max_sal, ig_like_sal, avg_sal = post_process_saliency(FinalData, all_gradients, scaling=scale_flag)
                saliency_dict[id] = ig_like_sal
                saliency_dict[id + 1] = max_sal
                id += 2
                print(f'Current ID: {id}')

        print(len(saliency_dict))
        print(id)
        
        post_processed_maps.append(saliency_dict)
        
        if viz:
            plot_processed_maps(FinalData, Labels, saliency_dict, Dataset[data], titles, p)
        
    return post_processed_maps, titles


# For the pretrained (on imagenet) models
def post_process_maps(data, fname, method_list=None, random_seeds=None, viz=True, scale_flag=False):
    
    
    Dataset = {0: 'mnist', 1: 'fmnist', 2: 'cifar10', 3: 'imgnet', 4: 'imgnet_bgc'}
    
    DatasetDict = {"mnist": load_mnist_saliency_data, "fmnist": load_fmnist_saliency_data,
                   'cifar10': load_cifar10_saliency_data, 'imgnet': load_imgnet_val_data, 
                   'imgnet_bgc': load_bgc_imagenet_saliency_data}
    
    if Dataset[data] == 'imgnet':
        raw_images,  FinalData, Labels, dataLoaderSal, classes = DatasetDict[Dataset[data]]()
    else:
        FinalData, Labels, dataLoaderSal, classes = DatasetDict[Dataset[data]]()
    print('Final Data Shape:', FinalData.shape)

    print('Test Data Shape:', Labels.shape)

    
    prefix_gd = fname+'_Grad_zero_'
    prefix_gd_input = fname+'_InputXGrad_zero_'
    prefix_ig = fname+'_IG_zero_'
    prefix_capt_ig = fname+'_CaptumIG_zero_'
    prefix_local_ig= fname+'_LocalIGAll_zero_'
    
#     prefix_only_ig= fname+'_LocalIGAll_zero_only_ig_'
#     prefix_only_max= fname+'_LocalIGAll_zero_only_max_'
#     prefix_path_grad= fname+'_LocalIGAll_zero_path_grad_'
    
    prefix_only_ig= fname+'_only_ig_zero_'
    prefix_only_max= fname+'_only_max_zero_'
    prefix_path_grad= fname+'_path_grad_zero_'
    
    prefix_sgig = fname+'_SGIG_zero_'
    prefix_sgsqig = fname+'_SGSQIG_zero_'
    prefix_vgig = fname+'_VGIG_zero_'
    
#     prefix_gradasc =fname+'_GradAsc_iter_200_lr_0.0001_zero_'
#     prefix_multigradasc = fname+'_MultiGradAsc_iter_200_lr_0.0001_zero_'
#     prefix_wt = fname+'_WtPerturbation_iter_200_zero_'
    
    prefix_gradasc =fname+'_GradAsc_zero_'
    prefix_multigradasc = fname+'_MultiGradAsc_zero_'
    prefix_wt = fname+'_WtPerturbation_zero_'
    
    prefix_gbp = fname+'_GBP_zero_'
    prefix_dlift= fname+'_DLift_zero_'
    prefix_lrp = fname+'_LRP_zero_'
    prefix_guidgradCam= fname+'_GuiGradCAM_zero_'


    all_methods = {"GD": prefix_gd, "Inp X GD": prefix_gd_input, "IG": prefix_ig, "CaptIG": prefix_capt_ig, \
                  "L.IG": prefix_local_ig, "ONLY.IG": prefix_only_ig, "ONLY.M": prefix_only_max, "SGIG": prefix_sgig, \
                  "SGSQIG": prefix_sgsqig, "VGIG": prefix_vgig, "GDAsc": prefix_gradasc, "M.GDAsc": prefix_multigradasc, \
                   "Wt.P": prefix_wt, "GBP": prefix_gbp, "DL": prefix_dlift, "LRP": prefix_lrp, "GradCAM": prefix_guidgradCam}

    filenamepath = "./models and saliencies"
    filenamepath = os.path.join(filenamepath, 'saliency')
    
    filenames, titles = [], []
    
    for method in method_list:
        filenames.append(all_methods[method])
        print(filenames[-1])
        
        if method not in ["GDAsc", "M.GDAsc", "Wt.P"]:
            titles.extend([method])
        else:
            titles.extend([method+'.IG', method+'.M'])
        
    print("Titles for Chosen Methods: ", titles)

    post_processed_maps = []
    saliency_dict = {}

    for p in random_seeds:

        id = 0

        print("Processing of gradients files started...")
        for fname in filenames:
            fname = os.path.join(filenamepath, fname+str(p)+'.npy')
            print(fname)
            all_gradients = np.load(fname)
           

            all_gradients = np.nan_to_num(all_gradients)
#             all_gradients = np.abs(all_gradients)

            print(all_gradients.shape)


            array_has_nan = np.isnan(all_gradients).any()
            input_has_nan = np.isnan(FinalData).any()
            print('Grads have NaNs:', array_has_nan)
            print('Inputs have NaNs:', input_has_nan)


            if "Grad" in fname and not 'XGrad' in fname and not "CAM" in fname and \
                len(all_gradients.shape) == FinalData.squeeze().dim():
                
                saliency_dict[id] = all_gradients
                id += 1
                print(f'Current ID: {id}')
                
            elif len(all_gradients.shape) == FinalData.squeeze().dim():
                saliency_dict[id] = all_gradients
                id += 1
                print(f'Current ID: {id}')

            elif "MultiDir" in fname and len(all_gradients.shape) == FinalData.squeeze().dim() + 1:
                max_sal, ig_like_sal, avg_sal = post_process_saliency(FinalData, all_gradients, scaling=scale_flag)
                saliency_dict[id] = max_sal
                saliency_dict[id + 1] = ig_like_sal
                saliency_dict[id + 2] = avg_sal
                id += 3
                print(f'Current ID: {id}')
            elif len(all_gradients.shape) == FinalData.squeeze().dim() + 1:
                max_sal, ig_like_sal, avg_sal = post_process_saliency(FinalData, all_gradients, scaling=scale_flag)
                saliency_dict[id] = ig_like_sal
                saliency_dict[id + 1] = max_sal
                id += 2
                print(f'Current ID: {id}')

        print(len(saliency_dict))
        print(id)
        
        post_processed_maps.append(saliency_dict)
        
        if viz:
            plot_processed_maps(FinalData, Labels, saliency_dict, Dataset[data], titles, p)
        
    return post_processed_maps, titles

def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
           
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


def compute_accuracy(outputs, y):
    _, preds = torch.max(outputs.data, 1)
    accuracy = (preds == y).sum().item()
    accuracy /= y.size(0)
    return accuracy

