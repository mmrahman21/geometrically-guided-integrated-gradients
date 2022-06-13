
"""
Preparing this file after submission to PNAS: This was not used for whole MILC + RAR work. 
For that, see milc_utilities_handler.py
"""


import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd import Variable
from torch.autograd.functional import jvp
from torch.autograd.functional import jacobian
import copy

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    ShapleyValues,
    ShapleyValueSampling,
    Lime,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Occlusion, 
    Saliency,
    GuidedBackprop,
)

def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:])/2.0
    integrated_gradients = torch.mean(grads, 0)
    return integrated_gradients

def compute_gradients(model, device, images, target_class_idx):
    images = Variable(images, requires_grad=True)
    probs = model(images)
#     print(probs.shape)
    scores = probs[:, target_class_idx]
#     print(scores.shape)
    grad_outputs = torch.zeros(probs.shape).to(device)
    grad_outputs[:, target_class_idx] = 1
    gradients = grad(outputs=probs, inputs=images, grad_outputs=grad_outputs, retain_graph=False)[0]
    
    return gradients

def construct_interpolate_images(baseline,
                       image,
                       alphas):
    alphas_x = alphas
    for i in range(len(image.shape)-1):
        alphas_x = alphas_x.unsqueeze(1)
        
    delta = image - baseline 
    images = baseline + alphas_x * delta 

    return images


def get_scale_max_ig_saliency(model, loaderSal, baseline, device, m_steps=50, pred_labels=None):
    
    '''
    Compute IG-like saliency, but take max only, not integral
    '''
    
#     model.train()

    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    saliencies = []
    
    alphas = torch.linspace(start=0.0, end=1.0, steps=m_steps+1).to(device)
    
    for i, data in enumerate(loaderSal):
        if i % 50 == 0:
            print("Processing subject: {}".format(i))
        x, y = data 
        
        x = x.to(device)
        y = y.to(device)
        
        if not pred_labels:
            
            out = model(x)
            _, preds = torch.max(out.data, 1)
            
        else:
            
            preds = pred_labels[i]
        
        interp_images = construct_interpolate_images(baseline=baseline,
                                   image=x,
                                   alphas=alphas)

        path_gradients = compute_gradients(model, device, 
        images=interp_images,
        target_class_idx=preds)

        max_grad = torch.max(path_gradients, 0)[0]
        final_max_grad = (x - baseline)*max_grad
        
        print('Input and Baseline Shape:', x.shape, baseline.shape)
        print("Path Gradients Shape:", path_gradients.shape)
        print('Max Grads:', max_grad.shape)
        print('Final Max Grad Shape:', final_max_grad.shape)
        
        saliencies.append(np.squeeze(final_max_grad.cpu().detach().numpy()))
        
    stacked_saliencies = np.stack(saliencies, axis=0)
    print(stacked_saliencies.shape)
             
    return stacked_saliencies

def get_local_ig_saliency(model, loaderSal, baseline, device, m_steps=50, pred_labels=None):
    
    '''
    Compute IG locally/from scratch
    '''
    
#     model.train()
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    saliencies = []
    ig_saliencies_not_scaled = []
    max_saliencies_not_scaled = []
    all_path_gradients = []
    
    alphas = torch.linspace(start=0.0, end=1.0, steps=m_steps+1).to(device)
    
    for i, data in enumerate(loaderSal):
        if i % 50 == 0:
            print("Processing subject: {}".format(i))
        x, y = data 
        
        x = x.to(device)
        y = y.to(device)
        
        if not pred_labels:
            
            out = model(x)
            _, preds = torch.max(out.data, 1)
            
        else:
            
            preds = pred_labels[i]
        
        interp_images = construct_interpolate_images(baseline=baseline,
                                   image=x,
                                   alphas=alphas)

        path_gradients = compute_gradients(model, device, 
        images=interp_images,
        target_class_idx=preds)
        
        path_gradients = torch.nan_to_num(path_gradients)
        
        max_grad = torch.max(path_gradients, 0)[0]
        max_saliencies_not_scaled.append(np.squeeze(max_grad.cpu().detach().numpy()))
        
        print("Path Grads Shape:", path_gradients.shape)
        
        all_path_gradients.append(np.squeeze(path_gradients.cpu().detach().numpy()))

        print("Shape:", path_gradients.shape)

#         path_grads = np.squeeze(path_gradients.cpu().detach().numpy())
        
        ig = integral_approximation(path_gradients)
        
        ig_saliencies_not_scaled.append(np.squeeze(ig.cpu().detach().numpy()))
       
        final_ig = (x.cpu().detach().numpy())*(ig.cpu().detach().numpy())
        
        print('Input and Baseline Shape:', x.shape, baseline.shape)
        print("Path Gradients Shape:", path_gradients.shape)
        print("IG Shape:", ig.shape)
        print("Final IG Shape:", final_ig.shape)

        saliencies.append(np.squeeze(final_ig))
     
    stacked_saliencies = np.stack(saliencies, axis=0)
    print(stacked_saliencies.shape)
    stacked_path_gradients = np.stack(all_path_gradients, axis=0)
    stacked_ig_saliencies = np.stack(ig_saliencies_not_scaled, axis=0)
    stacked_max_saliencies = np.stack(max_saliencies_not_scaled, axis=0)
             
    return stacked_saliencies, stacked_path_gradients, stacked_ig_saliencies, stacked_max_saliencies

def get_captum_ig_saliency(model, loaderSal, baseline, device, n_steps=50, pred_labels=None):
    
    '''
    Use Captum for IG
    '''
    
#     model.train()
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    saliencies = []
    sal = IntegratedGradients(model)
    
    for i, data in enumerate(loaderSal):
        if i % 50 == 0:
            print("Processing subject: {}".format(i))
        x, y = data   
        
        x = x.to(device)
        y = y.to(device)
        
        if not pred_labels:
            
            out = model(x)
            _, preds = torch.max(out.data, 1)
            
        else:
            
            preds = pred_labels[i]
            
        
        x.requires_grad_()
        
        S = sal.attribute(x,baseline,target=preds, n_steps=n_steps)
        
        saliencies.append(np.squeeze(S.cpu().detach().numpy()))
        
    stacked_saliencies = np.stack(saliencies, axis=0)
    print(stacked_saliencies.shape)
             
    return stacked_saliencies

def get_grad_ascent_saliency(model, loaderSal, iterations, lr, device, pred_labels=None):
    
    '''
    Single gradient ascent method
    '''
    
#     model.train()
    
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    all_saliencies = defaultdict(list)
    
    print('Revised Gradient Ascent...')
    
    for i, data in enumerate(loaderSal):
        if i % 50 == 0:
            print("Processing subject: {}".format(i))
        x, y = data   
        
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_()
        
        saliencies = []
        
        if not pred_labels:
            out = model(x)
            _, preds = torch.max(out.data, 1)
            
        else:
            preds = pred_labels[i]
        
        for j in range(iterations):
            
            model.zero_grad()
            out = model(x)

            grad_outputs = torch.zeros(out.shape).to(device)
            grad_outputs[:, preds] = 1
            
            if i % 50 == 0:
                print(f"Logit Score: {out[:, y]}")
            
            gradients = grad(outputs=out, inputs=x, grad_outputs=grad_outputs, retain_graph=False)[0]
            
            #Normalized gradients -1 to +1
#             direction_vec = (gradients - torch.min(gradients))/(torch.max(gradients)-torch.min(gradients))
#             direction_vec = torch.nan_to_num(direction_vec)
#             direction_vec = 2*direction_vec - 1
            
#             direction_vec = direction_vec * (abs(direction_vec) > 0) + 0.01 * (abs(direction_vec) <= 0)
            
#             x.data = x.data + lr * direction_vec

            grads = gradients.detach()
    
#             if grads.norm() == 0:
#                 print(f'Peak Reached...at:{j}')
#                 break

            x.data = x.data + lr * grads
            
            saliencies.append(np.squeeze(gradients.cpu().detach().numpy()))  # saving grad for checking instead of accumulation
    
        saliencies = np.stack(saliencies, axis=0)
        print(saliencies.shape)
        all_saliencies[i] = saliencies
    
    all_saliency_result = list(all_saliencies.values())
    all_saliency_result = np.array(all_saliency_result)
    print("All Saliencies Combined:", all_saliency_result.shape)
    
    return all_saliency_result

 
def get_multi_grad_ascent_max_saliency(model, loaderSal, baseline, iterations, lr, device, m_steps=50, pred_labels=None):
    
    '''
    Implement Interpolated Gradient Ascent
    '''
    
#     model.train()

    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    all_saliencies = []
    
    print('Multi-point Gradient Ascent...')
    
    alphas = torch.linspace(start=0.0, end=1.0, steps=m_steps+1).to(device)
    
    for i, data in enumerate(loaderSal):
        if i % 50 == 0:
            print("Processing subject: {}".format(i))
        x, y = data  
        
        x = x.to(device)
        y = y.to(device)
        
        if not pred_labels:
            
            out = model(x)
            _, preds = torch.max(out.data, 1)
            
        else:
            preds = pred_labels[i]
       
        saliencies = []
        
        interp_images = construct_interpolate_images(baseline=baseline,
                                   image=x,
                                   alphas=alphas)
        
        # You got your interpolation images per sample
        # do grad ascent for each interpolation point
        
        for j in range(iterations):
            
            model.zero_grad()
            
            path_gradients = compute_gradients(model, device,
                images=interp_images,
                target_class_idx=preds)
                        
            #Normalized gradients -1 to +1
#             direction_vec = (path_gradients - torch.min(path_gradients))/(torch.max(path_gradients)-torch.min(path_gradients))
#             direction_vec = torch.nan_to_num(direction_vec)
#             direction_vec = 2*direction_vec - 1
            
#             direction_vec = direction_vec * (abs(direction_vec) > 0) + 0.01 * (abs(direction_vec) <= 0)
            
#             interp_images.data = interp_images.data + lr * direction_vec
            
            path_gradients = path_gradients.detach()
            interp_images.data = interp_images.data + lr * path_gradients
            
            saliencies.append(np.squeeze(path_gradients.cpu().detach().numpy()))  # saving grad for checking instead of accumulation
    
        saliencies = np.stack(saliencies, axis=0)
        print(saliencies.shape)
        
        max_saliencies_interp_images = torch.max(torch.from_numpy(saliencies), 0)[0]
        print(f"Subj {i}: {max_saliencies_interp_images.shape}")
        all_saliencies.append(max_saliencies_interp_images.numpy())
        
        
    all_saliency_result = np.stack(all_saliencies, axis=0)
    print("All Saliencies Combined:", all_saliency_result.shape)
    
    return all_saliency_result

def saliency_through_weight_perturbation(model, main_model_dict, loaderSal, iterations, device, pred_labels=None):
    
#     model.train()
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    all_saliencies = defaultdict(list)
    
    for i, data in enumerate(loaderSal):
        if i % 50 == 0:
            print("Processing subject: {}".format(i))
        x, y = data   
        print("Data Shape:", x.shape)
        
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_()
        
        saliencies = []
        
        counter = 0
        
        while counter < iterations:
            model.load_state_dict(main_model_dict)
            model.to(device)
            out_main = model(x)
            
            main_scores = F.softmax(out_main, dim=1)
            _, preds_main = torch.max(main_scores.data, 1)
            
            other_dict = copy.deepcopy(main_model_dict)
            for key, value in main_model_dict.items():

                if 'running' not in key and other_dict[key].dtype != torch.int64:
                    key_std = other_dict[key].std()
                    eps = torch.normal(0, 0.001, size=other_dict[key].shape).to(device)
                    other_dict[key] += eps
                
            model.load_state_dict(other_dict)
            model.to(device)
            
            model.zero_grad()
            out_perturb = model(x)
            
            perturb_scores = F.softmax(out_perturb, dim=1)
            _, preds_perturb = torch.max(perturb_scores.data, 1)
            
            if i % 50 == 0:
                print(f"Original Logit Score: {out_main[:, y]}, \nPerturb Logit Score: {out_perturb[:, y]}")
                print(f"\n\nOriginal Prob Score: {main_scores[:, y]}, \nPerturb Prob Score: {perturb_scores[:, y]}")
            
            if main_scores[:, y] - perturb_scores[:, y] <= 0 and preds_main == preds_perturb:
                
                print(f'Iteration: {counter} \nSUCCESS...THIS PERTURBATION IS GOOD')
                grad_outputs = torch.zeros(out_perturb.shape).to(device)
                
                if not pred_labels:
                    grad_outputs[:, preds_perturb] = 1
                    
                else:
                    grad_outputs[:, pred_labels[i]] = 1
                    
           
                gradients = grad(outputs=out_perturb, inputs=x, grad_outputs=grad_outputs, retain_graph=False)[0]
                saliencies.append(np.squeeze(gradients.cpu().detach().numpy()))  # saving all grads 
                
                counter += 1
                
            else:
                 print(f'Iteration: {counter} \nSORRY...SCORE IS NOT ENOUGH. TRY THIS PERTURBATION AGAIN.', counter)
    
        saliencies = np.stack(saliencies, axis=0)
        print('Sub {} completed, saliency shape {}'.format(i, saliencies.shape))
        all_saliencies[i] = saliencies
    
    all_saliency_result = list(all_saliencies.values())
    all_saliency_result = np.array(all_saliency_result)
    print("All Saliencies Combined:", all_saliency_result.shape)
    
    return all_saliency_result
    

def post_process_saliency(main_data, all_gradients, scaling:bool):
    
    '''
    post-process saliency for finalization and visualization
    '''
    
    print("Main Data shape:", main_data.shape)
    print("Grad Shape:", all_gradients.shape)

    max_saliencies = []
    ig_like_saliencies = []
    avg_saliencies = []
    
    print(f'Scaling? {scaling}')
    for grads, data in zip(all_gradients, main_data):

        ig = integral_approximation(torch.from_numpy(grads))
        
        if scaling:
            final_ig = data*(ig.numpy())
        else:
            final_ig = ig.numpy()
        
        ig_like_saliencies.append(np.squeeze(final_ig))

        max_grad = torch.max(torch.from_numpy(grads), 0)[0]
        if scaling:
            final_max_grad = data*(max_grad.numpy())
        else:
            final_max_grad = max_grad.numpy()
       
        max_saliencies.append(np.squeeze(final_max_grad))
        
        avg = torch.mean(torch.from_numpy(grads), 0)
        
        if scaling:
            final_avg = data*(avg.numpy())
        else:
            final_avg = avg.numpy()
            
        avg_saliencies.append(np.squeeze(final_avg))

    max_saliencies = np.stack(max_saliencies, axis=0)
    print("Post Process:", max_saliencies.shape)
    ig_like_saliencies = np.stack(ig_like_saliencies, axis=0)
    print("Post Process:", ig_like_saliencies.shape)
    avg_saliencies = np.stack(avg_saliencies, axis = 0)
    print("Post Process:", avg_saliencies.shape)
    
    return max_saliencies, ig_like_saliencies, avg_saliencies