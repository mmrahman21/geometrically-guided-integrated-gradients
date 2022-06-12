###############################################################################################################
#        This script is used to compute saliency (after finished training!) on real benchmark datasets        #
###############################################################################################################

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import copy

from myscripts.milc_utilities_handler import load_pretrain_model, get_captum_saliency, save_predictions, save_reload_model,save_acc_auc
from datetime import datetime
import pandas as pd

import sys
import os
import time
import gc

from dnn_utils import update_lr, schedule_lr_decay, train, test

import torch.optim as optim 
from models import ConvModel, VGGNet, LeNet, MNNet, SanityCheckCNN, SanityCheckMLP, FashionCustomizedCNN
import argparse
import dataloader

from interpretability import ( 
    integral_approximation, compute_gradients, construct_interpolate_images, 
    get_scale_max_ig_saliency, get_local_ig_saliency, get_captum_ig_saliency, 
    get_grad_ascent_saliency, get_multi_grad_ascent_max_saliency, 
    get_multiple_direc_deriv_saliency, get_multiple_direc_deriv_saliency2,
    get_direct_vec_saliency, saliency_through_weight_perturbation, 
    get_extended_vec_saliency
)
    
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    PerturbationAttribution,
    ShapleyValues,
    ShapleyValueSampling,
    KernelShap,
    Lime,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Occlusion, 
    Saliency,
    GuidedBackprop,
    LRP,
    InputXGradient,
    Deconvolution,
    GuidedGradCam, 
)

parser = argparse.ArgumentParser(description='Saliency Computation')

parser.add_argument('--dataset', type=str, default=None, metavar='DATA',
                    help='dataset to use (default: None)')

parser.add_argument('--saliency_id', type=int, default=4, metavar='Sal ID',
                    help='saliency method id (default: 4)')

parser.add_argument('--iterations', type=int, default=200, metavar='ITER',
                    help='number of iterations to compute saliency (default: 200)')
parser.add_argument('--sal_lr', type=float, default=1.0, metavar='LR',
                    help='learning rate for iterative saliency computation (default: 1.0)')
parser.add_argument('--base', type=int, default=0, metavar='Baseline(Zero/Random/No)',
                    help='baseline to use (default: 0)')

parser.add_argument('--seeds', type=int, default=1, metavar='Random Seed',
                    help='models (default: 1)')

args = parser.parse_args()

print(args)

loaders, classes = dataloader.loader(args.dataset)

test_loader = loaders['test']
dataLoaderSal = loaders['saliency']


ensembles = {0:'', 1:'smoothgrad', 2:'smoothgrad_sq', 3: 'vargrad', 4:'', 5:'smoothgrad', 6:'smoothgrad_sq', 7: 'vargrad'}

saliency_options = {0:'Grad', 1:'SGGrad', 2: 'SGSQGrad', 3: 'VGGrad', 4: 'IG', 5: 'SGIG', 6:'SGSQIG', 7:'VGIG', 8: 'DLift', 9: 'DLiftShap', 10: 'ShapValSample', 11: 'ShapVal', 12: 'lime', 13: 'extendedVec', 14: 'GradAsc', 15: 'DirVec',
                    16: 'ScaleMaxIG', 17: 'LocalIGAll', 18: 'CaptumIG', 19: 'MultiDirVecM', 20: "MultiGradAsc", 
                    21: "WtPerturbation", 22: 'LRP', 23: 'GBP', 24: 'GuiGradCAM', 25: 'InputXGrad' }

baselines = { 0: 'zero', 1 : 'random', 2 : 'no_baseline'}

start_time = time.time()

device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

print(device)

prefix = "method_research_"+args.dataset+"_unsoftmaxed_vgg_customized_new" #+"_random_mod"
saliency_id = args.saliency_id
baseline_id = args.base
basename = os.path.join('models and saliencies', 'saliency')

inputs, classes = next(iter(test_loader)) 
print(inputs.shape)
sample = inputs[0]
print(sample.shape)

if baseline_id == 0:
    baseline = torch.zeros_like(sample).to(device)
    
elif baseline_id == 1:
    baseline = torch.randn_like(sample).to(device)
    
if baseline_id != 2:
    print("Baseline Shape", baseline.shape)
    
baseline = baseline.unsqueeze(0)
print("Baseline Shape", baseline.shape)

lr_sal = args.sal_lr
iterations = args.iterations
            
# Reload model 

# if args.dataset == "cifar10":
#     model = VGGNet(3, 10)
#     path = './models and saliencies/cifar_model_vgg_5.pth'
#     model_dict = torch.load(path, map_location=device)  # with good components
#     model.load_state_dict(model_dict)
#     model.to(device)
#     print('Model loaded from:', path)

# elif args.dataset == "mnist":
#     model = SanityCheckCNN()
#     path = './models and saliencies/method_research_mnist_model_'+str(model_id)+'.pth'
# #     path = './models and saliencies/mnist_random_model.pth'
#     model_dict = torch.load(path, map_location=device)  # with good components
#     model.load_state_dict(model_dict)
#     model.to(device)
#     print('Model loaded from:', path)
    
# elif args.dataset == "fmnist":
#     config = {}
#     model = ConvModel(config) 
# #     path = './models and saliencies/fashionmnist_model_3.pth'
#     path = './models and saliencies/fashion_mnist_model_5.pth'
# #     path = './models and saliencies/fashion_mnist_random_model.pth'
#     model_dict = torch.load(path, map_location=device)  # with good components
#     model.load_state_dict(model_dict)
#     model.to(device)
#     print('Model loaded from:', path)
    
# elif args.dataset == 'imgnet':
#     model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
#     model.to(device)
#     print('Inception Model Loaded...')

# for name, param in model.named_parameters():
#     print('Name: {} \n'.format(name))
    

criterion = nn.CrossEntropyLoss()

lr = None

start_time = time.time()

for qq in range(1):
    
    restart = args.seeds
    
#     model = SanityCheckCNN()

    model = VGGNet(3, 10) # CIFAR model

#     model = FashionCustomizedCNN()
    print(model)
    model.to(device)
    
    test(model, test_loader, criterion, device)
    
#     path = './models and saliencies/method_research_mnist_model_'+str(restart)+'.pth'
#     path = './models and saliencies/method_research_fmnist_model_vgg_customized_'+str(restart)+'.pth'
    path = './models and saliencies/method_research_cifar_model_vgg_customized_new_'+str(restart)+'.pth'
    
    model_dict = torch.load(path, map_location=device)  # with good components
    model.load_state_dict(model_dict)
    model.to(device)
    print('Model loaded from:', path)
    
    print(model)
    
    if saliency_options[saliency_id] == 'GuiGradCAM':
        lr = model.features[17]  # mnist: model.conv2
        print(lr)
    
    test(model, test_loader, criterion, device)

    # Quick Checks:

    if saliency_options[saliency_id] == "ScaleMaxIG":
        saliencies = get_scale_max_ig_saliency(model, dataLoaderSal, baseline, device, m_steps=50)
        print("Scaled Max IG Computed:", saliencies.shape)
        main_file = prefix+'_'+saliency_options[saliency_id]+'_'+baselines[baseline_id]

    elif saliency_options[saliency_id] == "LocalIGAll":
        saliencies, path_gradients, ig_saliencies, max_saliencies = get_local_ig_saliency(
            model, dataLoaderSal, baseline, device, m_steps=50)
        print("LocalIGAll Computed:", saliencies.shape)
        print("Path Gradients Computed:", path_gradients.shape)
        print("IG Gradients Computed:", ig_saliencies.shape)
        print("Max Gradients Computed:", max_saliencies.shape)
        main_file = prefix+'_'+saliency_options[saliency_id]+'_'+baselines[baseline_id]

        path_grad_file = prefix+'_'+saliency_options[saliency_id]+'_'+baselines[baseline_id]+'_path_grad'
        ig_file = prefix+'_'+saliency_options[saliency_id]+'_'+baselines[baseline_id]+'_only_ig'
        max_file = prefix+'_'+saliency_options[saliency_id]+'_'+baselines[baseline_id]+'_only_max'

        path_grad_path = os.path.join(basename, path_grad_file+'_'+str(restart))
        np.save(path_grad_path, path_gradients)
        print('path grads saved:', path_grad_path)

        ig_grad_path = os.path.join(basename, ig_file+'_'+str(restart))
        np.save(ig_grad_path, ig_saliencies)
        print('Only IG grads saved:', ig_grad_path)

        max_grad_path = os.path.join(basename, max_file+'_'+str(restart))
        np.save(max_grad_path, max_saliencies)
        print('Only max grad saved:', max_grad_path)

    elif saliency_options[saliency_id] == "CaptumIG":
        saliencies = get_captum_ig_saliency(model, dataLoaderSal, baseline, device, n_steps=50)
        print("Captum IG Computed:", saliencies.shape)
        main_file = prefix+'_'+saliency_options[saliency_id]+'_'+baselines[baseline_id]

    elif saliency_options[saliency_id] == "extendedVec":
        saliencies = get_extended_vec_saliency(model, dataLoaderSal, baseline, iterations, lr_sal, device)
        main_file = prefix+'_'+saliency_options[saliency_id]+'_iter_'+str(iterations)+'_lr_'+str(lr_sal)+'_'+baselines[baseline_id]

    elif saliency_options[saliency_id] == "DirVec":
        saliencies = get_direct_vec_saliency(model, dataLoaderSal, iterations, lr_sal, device)
        main_file = prefix+'_'+saliency_options[saliency_id]+'_iter_'+str(iterations)+'_lr_'+str(lr_sal)+'_'+baselines[baseline_id]

    # Gradient Ascent
    elif saliency_options[saliency_id] == "GradAsc":
        saliencies = get_grad_ascent_saliency(model, dataLoaderSal, iterations, lr_sal, device)
        print('New logit to input computed:', saliencies.shape)
        main_file = prefix+'_'+saliency_options[saliency_id]+'_iter_'+str(iterations)+'_lr_'+str(lr_sal)+'_'+baselines[baseline_id]

    # Multi-point Gradient Ascent
    elif saliency_options[saliency_id] == "MultiGradAsc":
        saliencies = get_multi_grad_ascent_max_saliency(model, dataLoaderSal, baseline, iterations, lr_sal, device, m_steps=50)
        print('New Multi-point Grad Ascent computed:', saliencies.shape)
        main_file = prefix+'_'+saliency_options[saliency_id]+'_iter_'+str(iterations)+'_lr_'+str(lr_sal)+'_'+baselines[baseline_id]

    elif saliency_options[saliency_id] == "WtPerturbation":
        saliencies = saliency_through_weight_perturbation(model, model_dict, dataLoaderSal, iterations, device)
        print('Wt Perturbation computed:', saliencies.shape)
        main_file = prefix+'_'+saliency_options[saliency_id]+'_iter_'+str(iterations)+'_'+baselines[baseline_id]

    elif saliency_options[saliency_id] == "MultiDirVecM":
        saliencies = get_multiple_direc_deriv_saliency(model, dataLoaderSal, 50, iterations, lr_sal, device)
        print('New multiple directional vector saliency...')
        main_file = prefix+'_'+saliency_options[saliency_id]+'_dirs_final_'+str(50)+'_iter_'+str(iterations)+'_lr_'+str(lr_sal)+'_'+baselines[baseline_id]

    else:
        saliencies = get_captum_saliency(model, dataLoaderSal, saliency_id , device, baseline = baselines[baseline_id], layer=lr)
        main_file = prefix+'_'+saliency_options[saliency_id]+'_'+baselines[baseline_id]

    sal_path = os.path.join(basename, main_file+'_'+str(restart))
    np.save(sal_path, saliencies)
    print("Saliency saved here:", sal_path)

elapsed_time = time.time() - start_time
print('Total Time Elapsed:', elapsed_time)
print(datetime.now())