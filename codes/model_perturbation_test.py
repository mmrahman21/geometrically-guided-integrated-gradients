import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F
import sys
from PIL import Image
from torchvision import transforms
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import os
from saliency_utilities import compute_saliency_for_methods
import copy
from datetime import datetime
import time
from methods.method_research_utilities import load_imagenet_saliency_data, load_imgnet_val_data
from torch import linalg as LA


class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.data = X
        self.label = Y
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y
    
    def __len__(self):
        return len(self.data)

def reinitialize_layers(copy_model, block_list):
    for name, param in copy_model.named_parameters():
        if "AuxLogits" not in name:
            for block in block_list:
                if block in name:
                    print("Re-initializing this parameter: {}".format(name))
                    if 'weight' in name or 'bias' in name:
                        nn.init.trunc_normal_(param, mean=0, std=0.1)
        else:
            print("This layer ({}) is not reinitializable".format(name))
            
    return True 
                    

def inception_block_names():
    layer_randomization_order = ['normal',
                                 'fc',
                                 'Mixed_7c',
                                 'Mixed_7b',
                                 'Mixed_7a',
                                 'Mixed_6e',
                                 'Mixed_6d',
                                 'Mixed_6c',
                                 'Mixed_6b',
                                 'Mixed_6a',
                                 'Mixed_5d',
                                 'Mixed_5c',
                                 'Mixed_5b',
                                 'Conv2d_4a_3x3',
                                 'Conv2d_3b_1x1',
                                 'Conv2d_2b_3x3',
                                 'Conv2d_2a_3x3',
                                 'Conv2d_1a_3x3']
    
    return layer_randomization_order

def add_noise_to_model(imgs, model, threshold, device=None):
    
    print('Perturbing model...')
    
    main_model_dict = model.state_dict()
    
    model.to(device)
    out_main = model(imgs)

    main_scores = F.softmax(out_main, dim=1).detach().cpu()
    best_score, preds_main = torch.max(main_scores.data, 1)
    
    other_dict = copy.deepcopy(main_model_dict)
    for key, value in main_model_dict.items():

        if 'running' not in key and other_dict[key].dtype != torch.int64:
            key_std = other_dict[key].std()
            perturb_std = key_std * (threshold/100)
            eps = torch.normal(0, perturb_std, size=other_dict[key].shape).to(device)
            other_dict[key] += eps

    model.load_state_dict(other_dict)
    model.to(device)

    model.zero_grad()
    out_perturb = model(imgs)

    perturb_scores = F.softmax(out_perturb, dim=1).detach().cpu()
    perturb_score, preds_perturb = torch.max(perturb_scores.data, 1)
    
    print("Predictions after perturbation: {} {}".format(perturb_score, preds_perturb))
    
    perturbed_score_original_pred = perturb_scores.gather(1, preds_main.view(-1,1))
    perturbed_score_original_pred = torch.squeeze(perturbed_score_original_pred)
    
    print('Main/Perturbed Preds: \n{}\n{}'.format(preds_main, preds_perturb))
    print('Original Best/Perturbed Original Scores: \n{}\n{}'.format(best_score, perturbed_score_original_pred))
    
    return model, best_score, perturbed_score_original_pred 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

saliency_methods = {0:'Grad', 1:'SGGrad', 2: 'SGSQGrad', 3: 'VGGrad', 4: 'IG', 5: 'SGIG', 6:'SGSQIG', 7:'VGIG', 8: 'DLift', 9: 'DLiftShap', 10: 'ShapValSample', 11: 'ShapVal', 12: 'lime', 13: 'extendedVec', 14: 'GradAsc', 15: 'DirVec',
                    16: 'ScaleMaxIG', 17: 'LocalIGAll', 18: 'CaptumIG', 19: 'MultiDirVecM', 20: "MultiGradAsc", 
                    21: "WtPerturbation", 22: 'LRP', 23: 'GBP', 24: 'GuiGradCAM', 25: 'InputXGrad' }

pretrained_models = {'Resnet18': models.resnet18, 'Resnet34': models.resnet34, 'resnet50': models.resnet50,
                     'Resnet101': models.resnet101, 'inception_3': models.inception_v3}

MODELS = {0 : 'Resnet18', 1: 'Resnet34', 2: 'resnet50', 3: 'Resnet101', 4: 'inception_3'}

method = saliency_methods[int(sys.argv[1])]

model_id = 3 #int(sys.argv[2])
name = MODELS[model_id]

# Load model

print("Loading pre-trained", MODELS[model_id], "model")
main_model = pretrained_models[MODELS[model_id]](pretrained=True)
main_model.to(device)
print(name)
print(main_model)
main_model.eval()


# Load Data
images, normalized_images, target_labels, dataLoaderSal, categories = load_imgnet_val_data() 
x_batch, y_batch = next(iter(dataLoaderSal))
print(x_batch.shape, y_batch.shape)
        
# Track time
since = time.time()

img_ids = [111, 114, 115, 122, 193]
imgs = []
targets = []

for id in img_ids:
    imgs.append(normalized_images[id])
    targets.append(target_labels[id])
    
imgs = torch.stack(imgs, axis=0)
targets = torch.stack(targets, axis=0)

print('Dataset Shape {}, {}'.format(imgs.shape, targets.shape))

# Generate prediction labels
imgs = imgs.to(device)
targets = targets.to(device)

outputs = main_model(imgs)
_, preds = torch.max(outputs, 1)

print(preds.shape)

preds= preds.tolist()

for pred in preds:
#     pred = pred.item()

    print(categories[pred])

# Create baseline and set other params
baseline = torch.zeros(1, 3, 224, 224).to(device)

if method == 'WtPerturbation':
    iterations = 200
else:
    iterations = 100
    
lr_sal = 0.0001

sal_img_loader = torch.utils.data.DataLoader(MyDataset(imgs, targets), batch_size=1, shuffle=False)
# Generate randomization order and compute saliency
# layer_randomization_order = inception_block_names()

# for i, layer_name in enumerate(layer_randomization_order):

#     layer_name = layer_randomization_order[i]
#     layer_list = layer_randomization_order[:i+1]
#     print("Going to randomize these layers {}".format(layer_list))
#     prefix = 'method_research_inception_revised_randomization_test_'+layer_name
#     copy_model = copy.deepcopy(main_model)

#     reinitialize_layers(copy_model, layer_list)

all_perturbation_soft_scores = []

for threshold in np.linspace(1, 100, 100):
    
    prefix = 'method_research_'+name+'_perturbation_test_'+ str(int(threshold))

    copy_model = copy.deepcopy(main_model)
    
    copy_model, best_score, perturbed_score_original_pred = add_noise_to_model(imgs, copy_model, threshold, device=device)
    
    if threshold == 1:
        all_perturbation_soft_scores.append(np.array(best_score))
    all_perturbation_soft_scores.append(np.array(perturbed_score_original_pred))
        
    print("B/A Scores: \n{}\n{}".format(best_score, perturbed_score_original_pred))

    copy_model_dict = copy.deepcopy(copy_model.state_dict()) 


    out = compute_saliency_for_methods(
            copy_model,
            copy_model_dict,
            sal_img_loader,
            method,
            baseline,
            iterations, 
            lr_sal,
            prefix,
            layer = None,
            steps = 50,
            device = device, 
            seed = 0,
            pred_labels = preds
        )

    print(len(out))

    basename = os.path.join('models and saliencies', 'saliency')

    output_details = {0: method, 1: 'path_grad', 2: 'only_ig', 3: 'only_max'}

    if method == 'LocalIGAll':

        for i in range(len(out)):
            main_file = prefix+'_'+output_details[i]+'_zero'
            sal_path = os.path.join(basename, main_file+'_'+str(0))
            np.save(sal_path, out[i])
            print("Output Shape: {}".format(out[i].shape))
            print("{} saved here {}".format(output_details[i], sal_path))

    else:
        main_file = prefix+'_'+method+'_zero'
        sal_path = os.path.join(basename, main_file+'_'+str(0))
        np.save(sal_path, out)
        print("Output Shape: {}".format(out.shape))
        print("{} saved here {}".format(method, sal_path))

comparative_prob_scores = np.stack(all_perturbation_soft_scores, axis=0)
print(comparative_prob_scores.shape)
score_path = os.path.join(basename, 'method_research_perturbation_test_all_scores_'+method+'_'+str(0))
np.save(score_path, comparative_prob_scores)
print("All comparative softmax scores saved here: {}".format(score_path))

elapsed_time = time.time() - since
print('Total Time Elapsed:', elapsed_time)
print(datetime.now())