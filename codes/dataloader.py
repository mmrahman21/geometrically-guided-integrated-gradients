import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch import nn
import os
from PIL import Image
from models import get_data_loader
from myscripts.generateSynData import artificial_batching_patterned_space1, artificial_batching_patterned_space2, artificial_batching_trend, multi_way_data, my_multi_block_var_data, actual_spatial, three_class, my_uniform_top_down, generate_top_down_pos_neg, generate_top_down_neg_neg, generate_top_down_pos_pos, generate_top_down_one_zero, generate_top_down_zero
import numpy as np


def loader(dataset):
    
    if dataset == 'mnist':
        trainX, trainY = torch.load('./my_mnist/MNIST/processed/training.pt')
        testX, testY = torch.load('./my_mnist/MNIST/processed/test.pt')
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
                                         (0.1307,), (0.3081,))
                                     ])

        trainX = transform(trainX)
        testX = transform(testX)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(trainX, trainY), 
            batch_size=batch_size_train, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(testX, testY), 
            batch_size=batch_size_test, shuffle=True)
        dataLoaderSal = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(testX[9000:] , testY[9000:]), 
            batch_size=1, shuffle=False)
        
        classes = ('zero', 'one', 'two', 'three',
           'four', 'five', 'six', 'seven', 'eight', 'nine')
        
    elif dataset == 'fmnist':
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
        
#         trainX = torch.clamp(trainX, -1, 1)
#         trainX = (trainX + 1)/2.0
#         testX = torch.clamp(testX, -1, 1)
#         testX = (testX + 1) / 2.0

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(trainX, trainY), 
            batch_size=batch_size_train, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(testX, testY), 
            batch_size=batch_size_test, shuffle=True)
        dataLoaderSal = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(testX[9500:] , testY[9500:]), 
            batch_size=1, shuffle=False)

        classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
    
    elif dataset == 'cifar10':
        transform_train = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #zooms, change shear angles.
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
                               ])
 
        transform = transforms.Compose([transforms.Resize((32,32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
        training_dataset = datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform_train) 
        test_dataset = datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform)
 
        train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=100, shuffle=True) 
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 100, shuffle=False)

        saliency_segment = list(range(200))
        saliency_dataset = torch.utils.data.Subset(test_dataset, saliency_segment)

        dataLoaderSal = torch.utils.data.DataLoader(saliency_dataset, batch_size=1,
                                            shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    elif dataset == 'imgnet':
        
        target = torch.as_tensor(13) 

        data = []
        labels = []

        filename = './ImageNet/collection/my_junco.jpg'

        input_image = Image.open(filename)
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        print(input_batch.shape)

        data.append(input_tensor)
        labels.append(target)

        indices = [2, 4, 19, 20, 27, 40, 55, 64, 83, 94, 105, 145, 169, 172, 242, 250, 256, 259, 260, 263]

        basename = './ImageNet/test/ILSVRC2012_test_'
        for idx in range(len(indices)):
            fname = basename + '00000' + str(indices[idx]).zfill(3)+ ".JPEG"
#             print(fname)
            input_image = Image.open(fname)
            input_tensor = preprocess(input_image)
            data.append(input_tensor)
            labels.append(target)

        data = torch.stack(data, dim=0)
        labels = torch.stack(labels, dim=0)
        labels = labels.long()
        print(data.shape, labels.shape)
        
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, labels), batch_size = 1, shuffle=False)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, labels), batch_size = 1, shuffle=False)

        dataLoaderSal = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data, labels), batch_size = 1, shuffle=False)
        
        # Read the categories
        with open("imagenet_classes.txt", "r") as f:
            classes = [s.strip() for s in f.readlines()]
    
    return {'train': train_loader, 'test': test_loader, 'saliency': dataLoaderSal}, classes


def synthetic_data_loader(data):
    Dataset = {0: 'oldtwocls', 1: 'newtwocls', 2: 'threecls', 3: 'multiwaytwocls', 4: 'uniformtwocls', 5: 'topdownposneg', 6: 'topdownnegneg', 7: 'topdownzero', 8: 'topdownonezero', 9: 'topdownpospos'}

    DatasetDict = {"oldtwocls": artificial_batching_patterned_space2, "newtwocls": actual_spatial, "threecls": three_class, "multiwaytwocls": multi_way_data, "uniformtwocls": my_uniform_top_down, "topdownposneg": generate_top_down_pos_neg, "topdownnegneg": generate_top_down_neg_neg, "topdownzero": generate_top_down_zero, "topdownonezero": generate_top_down_one_zero, 
                  "topdownpospos": generate_top_down_pos_pos}

    X, Y, start_positions, masks = DatasetDict[Dataset[data]](3000, 140, 50, seed=1988)
    print('Data Shape:', X.shape)
    X = np.moveaxis(X, 1, 2)  # it needs only for encoder

    print('Converted Data Shape:', X.shape)

    X_train = X[:2000]
    Y_train = Y[:2000]

    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).long()

    X_test = X[2500:]
    Y_test = Y[2500:]

    X_val = X[2000:2500]
    Y_val = Y[2000:2500]

    X_sal = X[2000:2500]
    Y_sal = Y[2000:2500]

    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).long()

    X_val = torch.from_numpy(X_val).float()
    Y_val = torch.from_numpy(Y_val).long()


    X_sal = torch.from_numpy(X_sal).float()
    Y_sal = torch.from_numpy(Y_sal).long()


    train_loader = get_data_loader(X_train, Y_train, 64)
    test_loader = get_data_loader(X_test, Y_test, X_test.shape[0])
#   val_loader = get_data_loader(X_val, Y_val, X_val.shape[0])
    sal_loader = get_data_loader(X_sal, Y_sal, 1)
    
    classes = ["top", "down"]
    
    return {'train': train_loader, 'test': test_loader, 'saliency': sal_loader}, classes



