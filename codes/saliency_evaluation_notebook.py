#!/usr/bin/env python
# coding: utf-8

# # Load all the required packages

# In[1]:


import pixellib
import cv2
from cv2 import GaussianBlur

import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import timeit
from scipy import interpolate
from scipy.stats import entropy

from sklearn import metrics
from functools import partial

from PIL import Image
from methods.saliency_recent_real_metrics import add_random_pixels, interpolate_missing_pixels,                 generate_saliency_focused_images_prev, generate_revised_saliency_focused_images, interpolate_img, calculate_webp_size

from methods.method_research_utilities import load_cifar10_saliency_data, post_process_maps
from methods.method_research_utilities import load_imagenet_saliency_data, load_imagenet_saliency_metric_eval_data, load_bgc_imagenet_saliency_data, load_cifar10_saliency_data, load_imgnet_val_data

from methods.captum_post_process import _normalize_image_attr

import os
import sys
import copy
from datetime import datetime
import time

from matplotlib.colors import LinearSegmentedColormap
from methods.saliency_utilities import plot_maps_method_vertical, plot_maps_method_horizontal

import io, os
import skimage.io
import skimage.filters
from skimage import color

from math import log, e
import seaborn as sns
import pandas as pd

# get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['svg']")
# get_ipython().run_line_magic('matplotlib', 'inline')


# ## Compute Sobel Edges

# In[2]:


def compute_sobel_edges(images):
    # Read the original image

    all_sobel_edges = []

    for (idx, img) in enumerate(images):
        
        if isinstance(img, np.ndarray):
            pass
        else:
            img = np.asarray(img).astype(np.uint8)

        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)
       
        # Sobel Edge Detection

        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

        all_sobel_edges.append(sobelxy)

    all_sobel_edges = np.stack(all_sobel_edges, axis=0)

    
    return all_sobel_edges


# ## Build Entropy-Softmax Interpolation Module
# 
# This module does interpolation to get common entropy-softmax values for all images. These common x-axis values will help to aggregate the result for the final PIC (SIC and AIC) plot. 

# In[3]:


def interpolate_entropy_vs_scores(relative_entropy, scores):

    f = interpolate.interp1d(relative_entropy, scores, fill_value="extrapolate")
    xnew = np.linspace(0.0, 1.0, 101)
    ynew = f(xnew) 
    return xnew, ynew


# ## Test the interpolation module

# In[ ]:


# x = np.linspace(0.0, 1.0, 10)
# y = np.exp(-x/3.0)
# print(interpolate_entropy_vs_scores(x, y))


# ## Prepare the images for entropy calculation. 
# 
# It requires unnormalized and original images. This module properly resizes images to $224 \times 224$ size and pixel values are kept in range $[0-255]$

# In[4]:


def get_unnormalized_images(images, target_labels):
    
    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()])

    unnormalized_images = []
    for image in images:
        if isinstance(image, np.ndarray) and image.shape[0]==image.shape[1]:
            img_tensor = torch.from_numpy(image)
            img_tensor = img_tensor.permute(2, 0, 1)
        else:
            img_tensor = transform(image)
            img_tensor = img_tensor*255
            
        img = img_tensor.to(int)
        unnormalized_images.append(img)

    unnormalized_images = torch.stack(unnormalized_images, dim=0)

    unnormalized_img_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(unnormalized_images, target_labels), batch_size = unnormalized_images.shape[0], shuffle=False)
    return unnormalized_images, unnormalized_img_loader
    


# # Normalize Images
# - argument images are already resized and within 0-255
# - output images are within 0-1 and z-scored

# In[5]:


def normalize_images(images, target_labels, samples_per_batch=None):
    
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    normalized_images = []
    labels = []
    
    for image, label in zip(images, target_labels):
        img_tensor = transform(image)
        normalized_images.append(img_tensor)
        labels.append(label)
        
    normalized_images = torch.stack(normalized_images, dim=0)
    labels = torch.stack(labels, dim=0)
    labels = labels.long()
    
    if samples_per_batch is None:
        normalized_img_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(normalized_images, labels), batch_size = normalized_images.shape[0], shuffle=False)
    
    else:
        normalized_img_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(normalized_images, labels), batch_size = samples_per_batch, shuffle=False)
        
    return normalized_images, normalized_img_loader


# ## Plot Images/Saliency Maps (for one method at a time)

# In[6]:


def plot_images_or_maps(images, labels=None, categories=None, nrows=3, ncols=3, samples_to_show=list(range(100)), plot_type='images', save=False):
    
    
    fig,axes=plt.subplots(nrows=nrows,ncols=ncols,figsize=(12,9),sharex=True,sharey=True)
    
    for i, ax in enumerate(axes.flat):
        
        img = images[samples_to_show[i]]
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        
        if img.shape[0] <=3:
            img = np.transpose(img, (1,2,0))
            
        if plot_type == 'images':
            ax.imshow(img, interpolation=None, aspect='auto')
        else:
            ax.imshow(img, cmap='Reds', vmin=0, vmax=1)
        
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

        if labels is not None:
            ax.set_title(categories[labels[i].item()], fontsize=6, y=0.95)
        
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()
    
    path = os.path.join('./Plots/Real/', "method_research_"+ plot_type)

    if save:
        fig.savefig(path+'.pdf', format='pdf', dpi=300)
        #     fig.savefig(path+'.svg', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
        print('Plots Saved...', path)
        
    plt.close(fig)


# ## Test the unnormalized but resized image generation module

# In[7]:


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
images, normalized_images, target_labels, dataLoaderSal, categories = load_imgnet_val_data() #load_imagenet_saliency_metric_eval_data()

unnormalized_images, unnormalized_img_loader = get_unnormalized_images(images, target_labels)
print(normalized_images.shape)
print(unnormalized_images.shape)
x_batch, y_batch = next(iter(unnormalized_img_loader))
image_sample = x_batch[2].numpy()
print(image_sample.shape)
print(np.min(image_sample), np.max(image_sample))

# plot_images_or_maps(x_batch, labels=target_labels, categories=categories, nrows=10, ncols=10)
# unnormalized_images = unnormalized_images.permute((0, 2, 3, 1))
# sobel_edges = compute_sobel_edges(unnormalized_images)
# sobel_edges = np.moveaxis(sobel_edges, 3, 1)

# print(sobel_edges.shape)
# print(np.min(sobel_edges[0]), np.max(sobel_edges[0]))
# plot_images_or_maps(sobel_edges, labels=target_labels, categories=categories, nrows=10, ncols=10)


# #### Set the computational device

# In[8]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Generate softmax scores
# 
# This function is to generate softmax scores on the **original** or the **saliency-focused** images

# In[26]:


def find_softmax_scores(model, normalized_images, pred_indices=None):
        
    with torch.no_grad():
        outputs = model(normalized_images)
        scores = F.softmax(outputs, dim=1)

        if pred_indices is None:
            best_prob_scores, pred_indices = torch.max(scores, dim=1)
        else:
            best_prob_scores = scores.gather(1, pred_indices.view(-1,1))
            best_prob_scores = torch.squeeze(best_prob_scores)
                
        return scores, best_prob_scores, pred_indices
    

class MyDataset(Dataset):
    def __init__(self, X):
        self.data = X
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)
    
def find_revised_softmax_scores(model, normalized_images, b_size=101, device=device):
    
    print("Original Data Shape:", normalized_images.shape)
    test_loader = torch.utils.data.DataLoader(MyDataset(normalized_images), batch_size=b_size, shuffle=False)
    
    all_scores = []
    all_best_prob_scores = []
    all_predictions = []
    with torch.no_grad():
        
        for (i, images) in enumerate(test_loader):
            
            if i % 10 == 0:
                print('Generating scores for {}-th batch'.format(i))
                print('Image Shape: {}'.format(images.shape))
            images = images.to(device)
            outputs = model(images)
            scores = F.softmax(outputs, dim=1).detach().cpu()
            all_scores.append(scores)
            
            best_prob_scores, pred_indices = torch.max(scores, dim=1)
            all_best_prob_scores.append(best_prob_scores)
            all_predictions.append(pred_indices)
            
    all_scores = torch.stack(all_scores, axis=0)
    all_scores = all_scores.squeeze(0)
    all_best_prob_scores = torch.stack(all_best_prob_scores, axis=0)
    all_best_prob_scores = all_best_prob_scores.squeeze(0)
    all_predictions = torch.stack(all_predictions, axis=0)
    all_predictions = all_predictions.squeeze(0)
    
    print(all_scores.shape)
    print(all_best_prob_scores.shape)
    print(all_predictions.shape)
        
    return all_scores.numpy(), all_best_prob_scores.numpy(), all_predictions
    


# ## Test the softmax score generation function as defined above
# 
# - Define the model
# - Load the weights
# - Generate the images
# - Call the softmax score generation function

# In[10]:


pretrained_models = {'Resnet18': models.resnet18, 'Resnet34': models.resnet34, 'resnet50': models.resnet50,
                     'Resnet101': models.resnet101, 'inception_3': models.inception_v3}

def load_pretrained_model(model_name):
    
    # Load the model
    print("Loading pre-trained", model_name, "model")
    model = pretrained_models[model_name](pretrained=True)
    return model


# In[21]:


# Load the images
# images, normalized_images, target_labels, dataLoaderSal, categories = load_imgnet_val_data() #load_imagenet_saliency_metric_eval_data()
# unnormalized_images, unnormalized_img_loader = get_unnormalized_images(images, target_labels)

# normalized_images = normalized_images.to(device)
# target_labels =target_labels.to(device)

# # Call the model loader
# model = load_pretrained_model('Resnet101')
# model.to(device)
# model.eval()

# # Call the softmax function to generate scores
# pred_indices = None
# scores, best_prob_scores, pred_indices = find_softmax_scores(model, normalized_images, pred_indices=pred_indices)
# print(categories[pred_indices[0].item()])
# print(scores.shape)
# print(best_prob_scores, pred_indices)


# ## post-process saliency maps

# In[ ]:


Dataset = {0: 'mnist', 1: 'fmnist', 2: 'cifar10', 3: 'imgnet'}
method_titles = ["GD", "ONLY.IG", "ONLY.M", "GDAsc.IG","GDAsc.M", "M.GDAsc.IG", "M.GDAsc.M","Wt.P.IG", "Wt.P.M", "IG", "CaptIG", "L.IG"]

def post_process_saliency_maps(dataset_id, model_name=None, methods=None, saliency_path_prefix=None):
    
    name = 'inception_3' if model_name is None else model_name
    methods_used = ["GD", "ONLY.IG", "ONLY.M", "GDAsc", "M.GDAsc", "Wt.P"] if methods is None else methods
    fname_common = "method_research_"+Dataset[dataset_id]+"_valSet_metricEval_"+name if saliency_path_prefix is None else saliency_path_prefix

    list_of_saliency_dicts, titles = post_process_maps(dataset_id, fname_common, method_list=methods_used,                                                          random_seeds=list(range(0, 1)), viz=False, scale_flag=False)
    return list_of_saliency_dicts[0], titles


# In[ ]:


# method_saliency_dict, title_set = post_process_saliency_maps(3, model_name='Resnet101') # arg: dataset_id


# ## Save and load processed saliency dictionary

# In[11]:



def change_keys_and_save_saliency_dict(method_saliency_dict, title_set, fname="process_saliency_for_metricEval"):
    
    saliency_dict_for_save = {}
    for key1,key2 in zip(method_saliency_dict.keys(), title_set):
        saliency_dict_for_save[key2] = method_saliency_dict[key1]

    np.savez(fname+".npz", **saliency_dict_for_save)

def load_saliency_dict_and_rename_keys(path=None):

    saliency_dict = np.load("process_saliency_for_metricEval.npz") if path is None else np.load(path)

    new_saliency_dict = {}

    for method_name, new_int_id in zip(saliency_dict.files, list(range(len(saliency_dict.files)))):
        new_saliency_dict[new_int_id] = saliency_dict[method_name]
    
    return new_saliency_dict, saliency_dict.files

# method_saliency_dict, title_set = load_saliency_dict_and_rename_keys()
# print(method_saliency_dict.keys())
# print(title_set)


# ## Do visualization of the comparable maps across several methods

# In[23]:


name = 'Resnet101_valSet_MetricEval'

def visualize_maps(name, display_range=[0, 10], f_size=(16, 16)):

    # "BlWhRd"
    colors = {'positive': 'Reds', 'absolute_value': 'Reds', 'all': LinearSegmentedColormap.from_list("RdWhGn", ["red", "white","green"])}
    sign = 'absolute_value'

    plot_maps_method_horizontal(model, normalized_images, target_labels, categories, name,                                 'imgnet', method_saliency_dict, title_set, 0,                                 range_to_display=np.asarray(range(display_range[0], display_range[1], 1)), fig_size=f_size, cm=colors[sign], vis_sign=sign)

    plot_maps_method_vertical(model, normalized_images, target_labels, categories, name,                                 'imgnet', method_saliency_dict, title_set, 0,                                 range_to_display=np.asarray(range(display_range[0], display_range[1], 1)), fig_size=f_size, cm=colors[sign], vis_sign=sign)
    


# visualize_maps(name, display_range=[100, 110], f_size=(16, 16))
    


# ## Normalize saliency maps 
# 
# Provide the processed saliency maps (not normalized and not channel collapsed) of all methods in a dictionary structure. This method returns a new dictionary of normalized maps for all methods. 
# 
# - The output saliency maps do not have channel dimension (i.e. now it is 2D for images)
# - attribution values are within 0-1 for absolute masks.

# In[12]:


def normalize_saliency_maps(saliency_method_dict, sign='absolute_value'):
    saliency_maps_all_method = {} 
       
    for method_name, all_saliency_images in saliency_method_dict.items():
        normalized_saliency = []
        for sal_image in all_saliency_images:
            attribution = np.transpose(sal_image, (1,2,0))
            norm_attr = _normalize_image_attr(attribution, sign)
           
            normalized_saliency.append(norm_attr)
        
        normalized_saliency = np.stack(normalized_saliency, axis=0)
       
        saliency_maps_all_method[method_name] = normalized_saliency
    return saliency_maps_all_method


# In[ ]:


# normalized_saliency_maps_all_method = normalize_saliency_maps(method_saliency_dict, sign='absolute_value')
# print(normalized_saliency_maps_all_method[0].shape)
# print(len(normalized_saliency_maps_all_method))

# sal_dict = {'edge': sobel_edges}
# print(sobel_edges.shape)
# normalized_edge_detector = normalize_saliency_maps(sal_dict, sign='absolute_value')
# print(normalized_edge_detector['edge'].shape)
# print(np.min(normalized_edge_detector['edge'][0]), np.max(normalized_edge_detector['edge'][0]))
# plot_images_or_maps(normalized_edge_detector['edge'], categories=categories, nrows=10, ncols=10, plot_type="metricEval_edge_detector", save=True)
# normalized_saliency_maps_all_method[9] = normalized_edge_detector['edge']
# title_set.append("Edge Detector")
# change_keys_and_save_saliency_dict(normalized_saliency_maps_all_method, title_set, fname="./MetricEvalEntropyMaps/normalized_saliency_for_valSet_metricEval_revised_with_edge_detector")


# ### Load from disk if the normalized saliency maps are already saved there

# In[13]:


normalized_maps_all_method, title_set = load_saliency_dict_and_rename_keys(path="./MetricEvalEntropyMaps/normalized_saliency_for_valSet_metricEval_revised_with_edge_detector.npz")
print(normalized_maps_all_method.keys())
print(title_set)


# ### Comparable visualization of normalized saliency masks across methods

# In[14]:


def visualize_comparable_saliency_maps(images, normalized_maps_all_method, method_names, samples_to_show=list(range(10)), save=False, fname=None):
    
    ncols, nrows = len(normalized_maps_all_method), 10
    
    fig,axes=plt.subplots(nrows=nrows,ncols=ncols + 1,figsize=(12,9),sharex=True,sharey=True)
    
    for row in range(nrows):
        
        img = images[samples_to_show[row]].numpy()
        
        if img.shape[0] <=3:
            img = np.transpose(img, (1,2,0))
            
        axes[row, 0].imshow(img, interpolation=None, aspect='auto')

    for method_id in range(ncols):

        saliency = normalized_maps_all_method[method_id]

        for i in range(nrows):

            sample = normalized_maps_all_method[method_id][samples_to_show[i]]
            
            if sample.shape[0] <= 3:
                sample = np.transpose(sample, (1, 2, 0))
            axes[i, method_id + 1].imshow(sample, interpolation=None, aspect='auto', cmap='Reds')

    #     Turn off *all* ticks & spines, not just the ones with colormaps.

            axes[i, method_id + 1].axes.xaxis.set_ticks([])
            axes[i, method_id + 1].axes.yaxis.set_ticks([])

        axes[0, method_id + 1].set_title(method_names[method_id], fontsize='medium')
        
    axes[0, 0].set_title('Image/Data', fontsize='medium')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
        
    plt.show()

    if save:
        path = os.path.join('./Plots/Real/', fname)
        fig.savefig(path+'.pdf', format='pdf', dpi=300)
        #     fig.savefig(path+'.svg', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
        print('Plots Saved...', path)
    plt.close(fig)


# In[ ]:


# method_names = ['GD', 'IG', 'IG_Max', 'GDA_IG', 'GDA_Max', 'GGIG_IG', 'GGIG_Max', 'GEG_IG', 'GEG_Max', 'Edge.D']
# visualize_comparable_saliency_maps(x_batch, normalized_maps_all_method, method_names, samples_to_show=[0, 10, 15, 24, 34, 44, 54, 64, 74, 80], save=True, fname="method_research_imgnet_test_inception_for_metricEval")


# ## Plot the comparative mask and blurred images for a group of methods

# In[15]:


def plot_comparable_saliency_focused_mages(saliency_methods, method_names, subj_id, after=False, save=False, fname_hint=None):
    
    ncols, nrows = len(saliency_methods), 10
    
    fig,axes=plt.subplots(nrows=nrows,ncols=ncols,figsize=(12,9),sharex=True,sharey=True)
    
    for method_id, method_name in zip(range(len(method_names)), method_names):

        saliency = saliency_methods[method_name]

        for i in range(nrows):

            sample = saliency[:, 1*after, :, :][5*i]
            
            if sample.shape[0] <= 3:
                sample = np.transpose(sample, (1, 2, 0))
            axes[i, method_id].imshow(sample, interpolation=None, aspect='auto', cmap=plt.cm.inferno)

    #     Turn off *all* ticks & spines, not just the ones with colormaps.

            axes[i, method_id].axes.xaxis.set_ticks([])
            axes[i, method_id].axes.yaxis.set_ticks([])

        axes[0, method_id].set_title(method_name, fontsize='medium')
        
    for interpolation_step in range(nrows):
        axes[interpolation_step, 0].set_ylabel("Threshold {}%".format(5*interpolation_step), fontsize=6)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
    if save:
        path = os.path.join('./Plots/Real/', "method_research_saliency_focused_images_"+fname_hint+"_"+str(after)+"_imageID_"+str(subj_id))
        fig.savefig(path+'.pdf', format='pdf', dpi=300)
    #     fig.savefig(path+'.svg', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
        print('Plots Saved...', path)
    plt.close(fig)


# ## Plot Relative Entropy vs. Threshold Information
# 
# - $x$ axis implies threshold (x% pixels added to the image)
# - $y$ asis gives relative entropy (entropy of linearly interpolated saliency image w.r.t original image)

# In[16]:


def plot_entropy_vs_threshold(percent_vs_entropy):
    
    fig = plt.figure()
    plt.plot(percent_vs_entropy[0], percent_vs_entropy[1])
    ax = plt.gca()
    ax.set_xlabel('Threshold %')
    ax.set_title('Threshold vs Entropy')
    ax.set_yticks(np.linspace(0.2, 1.0, num=9))
    ax.set_yticks(np.linspace(0.2, 1.0, num=9))
    ax.set_ylabel('Relative Entropy')

    plt.show()


# ## Create random image with 1% pixels and generate mask for absent pixels

# In[17]:


def create_random_img_and_mask_for_interpolation(numpy_image, initial_percentage=0.01):
    random_img, mask = add_random_pixels(numpy_image, p=initial_percentage)
    print(random_img.shape)
    print(np.count_nonzero(mask))
    mask_rand = (ma.array(mask[:, :]) == 0).data
    print("Mask Shape:", mask_rand.shape)
    return random_img, mask_rand


# ## Compute relative entropy (used for primary checking)

# In[ ]:


# numpy_image = np.transpose(x_batch[89].numpy(), (1,2,0))
# print(numpy_image.shape)
# print(np.min(numpy_image), np.max(numpy_image))

# main_size = calculate_webp_size(numpy_image)

# sal_mask = normalized_maps_all_method[6][89]

# # Create random image and interpolation mask mask
# random_img, mask_rand = create_random_img_and_mask_for_interpolation(numpy_image, initial_percentage=0.005)
# random_interpolated_img = interpolate_img(random_img, mask_rand, interp_mode='nearest')
# plt.imshow(random_interpolated_img)

# plt.xticks([])
# plt.yticks([])
# saliency_images, pixel_percent_n_entropy = generate_revised_saliency_focused_images(numpy_image, random_img, mask_rand, random_interpolated_img, \
#                              iterations=101, saliency_mask=sal_mask, interp_mode = 'nearest')

# print(saliency_images.shape)
# print(pixel_percent_n_entropy.shape)


# # Plot blurred images and corresponding entropy values

# In[ ]:


# print(saliency_images.shape)
# plot_images_or_maps(saliency_images[0:50, 1, :, :], nrows=5, ncols=5)
# plot_entropy_vs_threshold(pixel_percent_n_entropy)


# ## Execute evaluation
# 1. Start with an empty image (e.g. black image). Let's call it image A.
# 2. Add 1% of random pixels from the original image to image A.
# 3. According to a given saliency method, add top x% of the most important pixels from the original image to image A, where x% is a chosen threshold.
# 4. Apply linear interpolation (or nearest neighbors) to image A to find the value of pixels that were not added in step 2 or 3.
# 5. Compute model output on image A (let's call the value T). Compute model output on the original image (let's call the value U). Compute T/U and clip it to [0, 1]. Sometimes the intermediate image A can be assigned a higher score than the original image, thus unclipped T/U can be higher than 1.0. This step gives the y-axis value.
# 6. Compute the amount of information in image A (I) and the original image (J) using WebP. Compute I/J and clip it to [0, 1]. This will give you the x-axis value.
#     
#     $blurred\_image\_rel\_entropy = \frac{size\_of\_the\_blurred\_image \, - \, size\_of\_the\_random\_image}{size\_of\_the\_original\_image \, - \, size\_of\_the\_random\_image}$ 
#     
#     <code>Note from Author: Sorry, that was my mistake. The relative entropy value is calculated as ("image entropy" - "completely blurred image entropy") / ("original image entropy" - "completely blurred image entropy"). Thus, the saliency threshold 0% corresponds to a relative value equal to 0.</code>
#     
#     
# 7. Apply steps 3 to 6 on different thresholds (x% in step 3).
# 8. Apply steps 1 to 7 on different images and aggregate the result.
# 
#    <code>Note from Author: When computing the entropy and softmax we use 100 steps based on image area e.g. start with 1% of image -> 2% -> 3% until 100%. For each image we interpolate the normalized softmax to normalized entropy points between 0.0 and 1.0 (in x axis, you can use scipy interpolate.interp1d for this). For example points (entropy, softmax) = (0.1, 0.1), (0.5, 0.5), (1.0, 0.6) would produce a line with slope of 1 till 0.5 and slope of 0.2 afterwards. Then we take the median of these curves in y-axis over all the images. We use 100 interpolation points between 0.0 and 1.0 so that the x points are aligned across examples before we take the median in y-axis.</code> 

# #### **Step A:** Define a function to create blurred images for different threshold (x%) for all images, based on a saliency method

# #### Step A (Revision): to create blurred images 
# 
# - Use nearest neighbor interpolation once!
# - No interpolation after adding salient pixels
# - No use of compression package (main trouble). Rather directly use entropy

# In[18]:


## for each image, compute different blurred images, relevant entropy values per threshold (x%)
# Provide 

def create_new_set_of_blurred_images(all_images, all_images_saliency, sal_method_name):
    
    all_blurred_images = {}
    all_relative_entropy = {}
    
    since = time.time()
    
    for i, image in enumerate(all_images):
        print('Image Count: {}'.format(i))
        numpy_image = np.transpose(image.numpy(), (1,2,0))
        print(numpy_image.shape)
        print(np.min(numpy_image), np.max(numpy_image))

        # Create random image and interpolation mask mask
        random_img, mask_rand = create_random_img_and_mask_for_interpolation(numpy_image, initial_percentage=0.005)
        
        random_interpolated_img = interpolate_img(random_img, mask_rand, interp_mode='nearest')

        saliency_images, pixel_percent_n_entropy = generate_revised_saliency_focused_images(numpy_image, random_img, mask_rand, random_interpolated_img,                              iterations=101, saliency_mask=all_images_saliency[i], interp_mode = 'nearest')

        print(saliency_images.shape)
        
        print(pixel_percent_n_entropy.shape)
        all_blurred_images[str(i)] = saliency_images
        all_relative_entropy[str(i)] = pixel_percent_n_entropy
        elapsed_time = time.time() - since
        print('Total Time Elapsed:', elapsed_time)
    np.savez("./MetricEvalEntropyMaps/all_blurred_images_for_metricEval_May1_half_percent_"+sal_method_name+".npz", **all_blurred_images)
    np.savez("./MetricEvalEntropyMaps/all_blurred_images_rel_entropy_for_metricEval_May1_half_percent_"+sal_method_name+".npz", **all_relative_entropy)
    print(datetime.now())
    


# #### Step B:  Use the function defined above to create all sets of blurred images per saliency method

# In[ ]:


# method_id = 10 #int(sys.argv[1])
# if method_id <= 9:
#     all_images_saliency = normalized_maps_all_method[method_id]
#     sal_method_name = title_set[method_id]
#     print("Saliency Shape: {}".format(all_images_saliency.shape))

# else: 
#     all_images_saliency = [None]*len(x_batch)
#     sal_method_name = "Random"
#     print("This saliency does not have shape..It is all None...")
    
# print("Saliency Method Name: {}".format(sal_method_name))
# create_new_set_of_blurred_images(x_batch, all_images_saliency, sal_method_name)


# #### Step C (To Check): Visualizing the created blurred images and entropy trends

# In[ ]:


# title_set.append('Random')
# title_set = title_set[:10]
# print(title_set)
# method_id = 6
# sal_method_name = title_set[method_id]

# method_list = {'GD': "GD", 'ONLY.IG': "IG", 'ONLY.M': "IG_Max",                'GDAsc.IG': "GDA_IG", 'GDAsc.M': "GDA_Max", 'M.GDAsc.IG': "GGIG_IG",                'M.GDAsc.M': "GGIG_Max", 'Wt.P.IG': "GEG_IG", 'Wt.P.M': "GEG_Max",                'Random': "Random", "Edge Detector": "Edge Detector"}

# desired_methods = ['GD', 'ONLY.IG', 'M.GDAsc.IG', 'M.GDAsc.M', 'Wt.P.IG','Wt.P.M', 'Edge Detector', 'Random']

# subj_id = 95
# saliency_all_methods = {}
# methods = []
# for sal_method_name in desired_methods:
    
#     sal_method_new_name = method_list[sal_method_name]

#     blurred_images_before_n_after_interp = np.load("./MetricEvalEntropyMaps/all_blurred_images_for_metricEval_May1_half_percent_"+sal_method_name+".npz")

#     blurred_images_rel_entropy = np.load("./MetricEvalEntropyMaps/all_blurred_images_rel_entropy_for_metricEval_May1_half_percent_"+sal_method_name+".npz")

    
#     blurred_images = blurred_images_before_n_after_interp[str(subj_id)]
#     pixel_percent_rel_entropies = blurred_images_rel_entropy[str(subj_id)]
#     saliency_all_methods[sal_method_new_name]=blurred_images
#     methods.append(sal_method_new_name)
#     print(blurred_images.shape)
#     print(pixel_percent_rel_entropies.shape)

# matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# plot_comparable_saliency_focused_mages(saliency_all_methods, methods, subj_id, after=False, save=True, fname_hint="imgnet_valSet_Resnet101")

# # plot_saliency_focused_mages(blurred_images, plot_flag='before_interpolation')
# plot_saliency_focused_mages(blurred_images, plot_flag='after_interpolation')
# plot_entropy_vs_threshold(pixel_percent_rel_entropies)
# print(pixel_percent_rel_entropies)


# ## Some utility functions

# In[19]:


def compute_accuracy(outputs, y):
    _, preds = torch.max(outputs.data, 1)
    accuracy = (preds == y).sum().item()
    accuracy /= y.size(0)
    return accuracy

def compute_auc_for_pic_metric(x, y):
    
    measure_a = np.round(np.mean((y[:-1] + y[1:])/2.0, 0), 3)
    measure_b = np.round(metrics.auc(x, y), 3)
    assert measure_a == measure_b, "Both computations should produce same results."
    
    return measure_a


def prepare_dataframe_for_plot(all_method_result, legends):
    '''
    all_method_result should be in (samples, methods, interpolation_points) shape
    samples: how many images/samples
    methods: how many interpretability methods
    legends: name of the interpretability methods
    '''
    
    all_result_data_frames=[]

    for (idx, title) in enumerate(legends):
        
        print(all_method_result.shape)
        
        if idx == 7:
            print(all_method_result[:, idx, 100])
        
        per_method_intermediate_results = all_method_result[:, idx, 0::10]
        print(per_method_intermediate_results.shape)
        
        df = pd.DataFrame(per_method_intermediate_results)
        
        df.columns = np.linspace(0, 10, 11).astype(int).tolist()
        
        for time_interval in df.columns:
            per_interval_result = df[[time_interval]]
            per_interval_result.columns = ["Score"]
            per_interval_result.insert(0, "Threshold %", 10*time_interval)
            per_interval_result.insert(2, 'Method', title)
            all_result_data_frames.append(per_interval_result)

    FinalResult = pd.concat(all_result_data_frames, axis=0)
    FinalResult.reset_index(drop=True, inplace=True)
    
    return FinalResult


# #### **Step D :** Feed the blurred images per saliency method to produce scores 

# In[20]:


def generate_scores_on_saliency_focused_images(model, main_resized_unnormalized_images, main_labels, blurred_images, relative_entropies):
    print(main_resized_unnormalized_images.shape)
    
    main_images, main_img_loader = normalize_images(main_resized_unnormalized_images, main_labels)
    
    # Call the softmax function to generate scores
    all_scores, main_images_prob_scores, main_pred_indices = find_revised_softmax_scores(model, main_images, b_size=main_images.shape[0], device=device)
    
    all_images_all_interpolation = []
    all_images_all_relative_entropy = []
    
    for subj_id in range(main_images.shape[0]):
        
        if subj_id % 25 == 0:
            print("Working on Subj: {}".format(subj_id))
        subj_blurred_images = blurred_images[str(subj_id)]
        
        subj_blurred_images_after_interpolation = subj_blurred_images[:, 1, :, :, :]
#         print(subj_blurred_images_after_interpolation.shape)
#         subj_blurred_images_after_interpolation = np.moveaxis(subj_blurred_images_after_interpolation, 1, -1)
        pixel_percent_rel_entropies = relative_entropies[str(subj_id)]
        subj_blurred_images_after_interpolation = subj_blurred_images_after_interpolation.astype(np.uint8)
        entropies = pixel_percent_rel_entropies[1, :]
        
        subj_blur_images, subj_blur_img_loader = normalize_images(subj_blurred_images_after_interpolation, [main_pred_indices[subj_id]]*101)
        all_images_all_interpolation.append(subj_blur_images)
        all_images_all_relative_entropy.append(entropies)

    all_images_all_interpolation = torch.stack(all_images_all_interpolation, axis=0)
    all_images_all_relative_entropy = np.stack(all_images_all_relative_entropy, axis=0)
    print(all_images_all_interpolation.shape)
    print(all_images_all_relative_entropy.shape)
    all_images_all_interpolation = all_images_all_interpolation.reshape(-1, *all_images_all_interpolation.shape[2:]) 
    all_blur_images_all_prob_scores, best_blur_scores, blur_pred_indices = find_revised_softmax_scores(model, all_images_all_interpolation, b_size=101, device=device)

    return main_images_prob_scores, main_pred_indices, all_blur_images_all_prob_scores, best_blur_scores, blur_pred_indices, all_images_all_relative_entropy
    


# In[21]:


title_set.append('Random') # Add this to the title set
print(title_set)


# In[22]:



def generate_results_dictionary(main_image_batch, sal_method_name, model_name, blur_image_path=None, blur_image_rel_entropy_path=None):
    
    print("Saliency Method Name: {}".format(sal_method_name))

    if blur_image_path is None:
        blurred_images_before_n_after_interp = np.load("./MetricEvalEntropyMaps/all_blurred_images_for_metricEval_May1_half_percent_"+sal_method_name+".npz")
        blurred_images_rel_entropy = np.load("./MetricEvalEntropyMaps/all_blurred_images_rel_entropy_for_metricEval_May1_half_percent_"+sal_method_name+".npz")

    else:
        blurred_images_before_n_after_interp = np.load(blur_image_path+sal_method_name+".npz")
        blurred_images_rel_entropy = np.load(blur_image_rel_entropy_path+sal_method_name+".npz")

    # Call the model loader
    model = load_pretrained_model(model_name)
    model.to(device)
    model.eval()


    x_batch = main_image_batch.to(torch.uint8)
    main_images = x_batch.numpy()
    main_images = np.moveaxis(main_images, 1, -1)
    main_images_prob_scores, main_pred_indices, all_blur_images_all_prob_scores, best_blur_scores, blur_pred_indices, all_images_all_relative_entropy = generate_scores_on_saliency_focused_images(model, main_images, y_batch, blurred_images_before_n_after_interp, blurred_images_rel_entropy)

    print(main_images_prob_scores.shape)
    print(all_blur_images_all_prob_scores.shape)
    saliency_evaluation_result = {"Main Scores": main_images_prob_scores, "Main Predictions": main_pred_indices.numpy(), "All Blur Scores": all_blur_images_all_prob_scores, "Best Blur Scores": best_blur_scores, "Blur Preds": blur_pred_indices.numpy(), "All Blur Entropies": all_images_all_relative_entropy}
    
    np.savez("detail_saliency_evaluation_May2_half_percent_"+sal_method_name+".npz", **saliency_evaluation_result)
    
    return saliency_evaluation_result
#np.savez("detail_saliency_evaluation_Apr22_half_percent_"+sal_method_name+".npz", **saliency_evaluation_result)


# #### **Step E:** Define functions to finalize (Interpolate/aggregate) the scores (softmax or accuracy) vs. entropy

# In[23]:


def generate_interpolated_softmax_scores(all_blur_images_all_prob_scores,                                          all_images_all_relative_entropy, 
                                         main_pred_indices):
    
    all_common_interpolated_softmax_scores = []
    for subj_id in range(main_pred_indices.shape[0]):
        pred_indices=torch.LongTensor([main_pred_indices[subj_id]]*101)
        subj_scores = all_blur_images_all_prob_scores[subj_id].gather(1, pred_indices.view(-1,1))
        subj_scores = torch.squeeze(subj_scores)
        original_img_score = subj_scores.clone()[-1]
        
        subj_scores /= original_img_score


        rel_entropies = all_images_all_relative_entropy[subj_id].numpy()
        
        entropies, scores = interpolate_entropy_vs_scores(rel_entropies, subj_scores)
        all_common_interpolated_softmax_scores.append(scores)
        
    all_common_interpolated_softmax_scores = np.stack(all_common_interpolated_softmax_scores, axis=0)
    return all_common_interpolated_softmax_scores, np.median(all_common_interpolated_softmax_scores, axis=0)
        
        
        
def generate_accuracy_scores(all_images_all_relative_entropy, 
                                         main_pred_indices, blur_pred_indices):
    
    all_common_accuracy_scores = []
    all_entropies = []
    for threshold_id in range(blur_pred_indices.shape[1]):
        preds_per_info_level = blur_pred_indices[:, threshold_id]
        accuracy = (preds_per_info_level == main_pred_indices).sum().item()
        accuracy /= main_pred_indices.size(0)
        all_common_accuracy_scores.append(accuracy)
        
        entropy = torch.mean(all_images_all_relative_entropy[:, threshold_id]).item()
        all_entropies.append(entropy)
    
    entropies, all_common_interpolated_accuracy_scores = interpolate_entropy_vs_scores(all_entropies, all_common_accuracy_scores)
    
    return all_common_accuracy_scores, all_common_interpolated_accuracy_scores


# #### **Step F**: Evaluate and plot the SIC/AIC metric.

# In[27]:



pallette_1 = ['#b2182b','#ef8a62','#fddbc7','#f7f7f7','#d1e5f0','#67a9cf','#2166ac']
pallette_2 = ['#d73027','#f46d43','#fdae61','#fee090','#e0f3f8','#abd9e9','#74add1','#4575b4']
pallette_3 = ['#8c510a','#bf812d','#2166ac','#80cdc1','#35978f','#01665e', '#4575b4']

sns.set(style="whitegrid", font_scale=1.0)
# sns.set_palette(palette=pallette_3)

def compute_pic_and_plot(main_image_batch, opt_method, model_name, save=False, filehint=""):   # pic: performance information curve
    
    legend_list = ["GD", "IG", "IG_Max", "GDA_IG", "GDA_Max", "GGIG_IG", "GGIG_Max", "GEG_IG", "GEG_Max", "Edge Detector", "Random"]

    all_method_accuracy_results = []
    all_auc_scores = []
    all_method_all_scores = []

    used_legends = []
    
    fig = plt.figure()

    for method_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

        sal_method_name = title_set[method_id]
        used_legends.append(legend_list[method_id])
        
        saliency_evaluation_results = generate_results_dictionary(main_image_batch, sal_method_name, model_name)

#         saliency_evaluation_results = np.load("./MetricEvalEntropyMaps/detail_saliency_evaluation_Apr22_half_percent_"+sal_method_name+".npz")
#         if method_id == 0:
#             print(saliency_evaluation_results.files)
        main_images_prob_scores = torch.from_numpy(saliency_evaluation_results['Main Scores'])
        main_pred_indices = torch.from_numpy(saliency_evaluation_results['Main Predictions'])
        all_blur_images_all_prob_scores = torch.from_numpy(saliency_evaluation_results['All Blur Scores'])
        best_blur_scores = torch.from_numpy(saliency_evaluation_results['Best Blur Scores'])
        blur_pred_indices = torch.from_numpy(saliency_evaluation_results['Blur Preds'])
        all_images_all_relative_entropy = torch.from_numpy(saliency_evaluation_results['All Blur Entropies'])

        all_blur_images_all_prob_scores = all_blur_images_all_prob_scores.reshape(200, 101, -1)
        best_blur_scores = best_blur_scores.reshape(200, -1)
        blur_pred_indices = blur_pred_indices.reshape(200, -1)
        all_images_all_relative_entropy = all_images_all_relative_entropy.reshape(200, -1)

        evaluation_options = {'softmax' : partial(generate_interpolated_softmax_scores, all_blur_images_all_prob_scores, all_images_all_relative_entropy, main_pred_indices), 
                        'accuracy': partial(generate_accuracy_scores, all_images_all_relative_entropy, 
                                              main_pred_indices, blur_pred_indices)              
        }

        all_scores, normalized_scores = evaluation_options[opt_method]()
        
        per_method_evaluation_result = {"Normalized Scores": np.array(normalized_scores)}
    
        if save:
            result_save_path = "./MetricEvalEntropyMaps/final_eval_result_"+sal_method_name+"_"+opt_method+".npz"
            np.savez(result_save_path, **per_method_evaluation_result)
            print('Evaluation Result Saved for: {}\t Saved Here: {}'.format(sal_method_name, result_save_path))

#         per_method_eval_result = np.load("./MetricEvalEntropyMaps/final_eval_result_"+sal_method_name+"_"+opt_method+".npz")
#         normalized_scores = per_method_eval_result['Normalized Scores']
    
        all_method_accuracy_results.append(normalized_scores) 
        plt.plot(np.linspace(0, 1.0, 11), normalized_scores[0::10])
        auc = compute_auc_for_pic_metric(np.linspace(0.0, 1.0, 101), np.array(normalized_scores))
        all_auc_scores.append(auc)
        all_method_all_scores.append(all_scores)
        
    all_auc_scores = np.stack(all_auc_scores, axis=0)
    all_method_all_scores = np.stack(all_method_all_scores, axis=1)
    
    plt.legend(used_legends)
    plt.xlabel("Normalized Estimation of Entropy")
    
    y_label = "Median of Normalized "+opt_method.capitalize() if opt_method == 'softmax' else opt_method.capitalize()
    plt.ylabel(y_label)
    plt.show()
    
    if save:
        path = os.path.join('./Plots/Real/', "method_research_"+opt_method+"_"+filehint)
        print(path)
        fig.savefig(path+'.svg', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
        fig.savefig(path+'.pdf', format='pdf', dpi=300)
        print('Plots Saved...', path)
        
    plt.close(fig)
    
    return used_legends, all_auc_scores

#     return used_legends, all_method_all_scores, all_auc_scores


# #### Evaluate softmax information curve (SIC) and plot the results

# In[ ]:


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# plt.style.use("dark_background")
sns.set(style="whitegrid", font_scale=1.0)
# used_legends, all_scores, all_auc_scores = compute_pic_and_plot(x_batch, "softmax", "Resnet101", save=True, filehint="ResNet101_Val_SIC")
used_legends, all_auc_scores = compute_pic_and_plot(x_batch, "accuracy", "Resnet101", save=False, filehint="ResNet101_val_AIC_checking_code")
print(all_auc_scores)
print(used_legends)


# #### Process and plot boxes

# In[ ]:


# matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# df = prepare_dataframe_for_plot(all_scores, used_legends)
# print(df.shape)

# sns.set(style="whitegrid", font_scale=1.0)
# f = plt.figure(figsize=(8,6))

# seven_class = ['#b2182b','#ef8a62','#fddbc7','#f7f7f7','#d1e5f0','#67a9cf','#2166ac']
# eight_class = ['#d73027','#f46d43','#fdae61','#fee090','#e0f3f8','#abd9e9','#74add1','#4575b4']

# ax= sns.boxplot(x="Threshold %", y="Score",linewidth=1, width=0.7, hue="Method", data=df)
# ax.set_ylim([0.0, 2.5])
# ax.set_ylabel('Normalized Softmax Score')
# plt.legend(loc='upper left')
# path = os.path.join('./Plots/Real/', "method_research_normalized_softmax_all_scores" )
# print(path)
# f.savefig(path+'.svg', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
# f.savefig(path+'.pdf', format='pdf', dpi=300)
# print('Plots Saved...', path)
# plt.show()
# plt.close(f)


# #### Evaluate accuracy information curve (AIC) and plot the results

# In[ ]:


# plt.style.use("dark_background")
# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# print(title_set)
# sns.set(style="whitegrid", font_scale=1.0)
# used_legends, all_scores, scores = compute_pic_and_plot("accuracy")
# print(scores)


# ### Bar plots of information curve (AUC)

# In[ ]:


# matplotlib.rcParams.update(matplotlib.rcParamsDefault)
# f = plt.figure(figsize=(10,6))
# plt.bar(used_legends, scores, color ='maroon',width = 0.4)
# path = os.path.join('./Plots/Real/', "method_research_normalized_accuracy_scores_bar" )
# print(path)
# f.savefig(path+'.svg', transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
# f.savefig(path+'.pdf', format='pdf', dpi=300)
# print('Plots Saved...', path)
# plt.show()
# plt.close(f)

