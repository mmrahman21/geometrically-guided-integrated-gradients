import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

from interpretability import ( 
    get_scale_max_ig_saliency, get_local_ig_saliency, get_captum_ig_saliency, 
    get_grad_ascent_saliency, get_multi_grad_ascent_max_saliency, 
    get_multiple_direc_deriv_saliency, get_multiple_direc_deriv_saliency2,
    get_direct_vec_saliency, saliency_through_weight_perturbation, 
)

# for captum viz
import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap

from functools import partial
from myscripts.milc_utilities_handler import get_captum_saliency 

basename = os.path.join('models and saliencies', 'saliency')

saliency_methods = {0:'Grad', 1:'SGGrad', 2: 'SGSQGrad', 3: 'VGGrad', 4: 'IG', 5: 'SGIG', 6:'SGSQIG', 7:'VGIG', 8: 'DLift', 9: 'DLiftShap', 10: 'ShapValSample', 11: 'ShapVal', 12: 'lime', 13: 'extendedVec', 14: 'GradAsc', 15: 'DirVec',
                    16: 'ScaleMaxIG', 17: 'LocalIGAll', 18: 'CaptumIG', 19: 'MultiDirVecM', 20: "MultiGradAsc", 
                    21: "WtPerturbation", 22: 'LRP', 23: 'GBP', 24: 'GuiGradCAM', 25: 'InputXGrad' }


def compute_saliency_for_methods(
        model,
        model_dict,
        dataLoaderSal,
        method,
        baseline,
        iterations: int, 
        lr_sal: float,
        prefix: str,
        layer = None,
        steps = 50,
        device = 'cpu', 
        seed = 0,
        pred_labels = None
    ) -> None:
    
    saliency_id = list(saliency_methods.keys())[list(saliency_methods.values()).index(method)]
    
    print("Interpretability Method: {}".format(saliency_methods[saliency_id]))
    
    
    saliency_options = {
                        ('Grad', 'SGGrad', 'SGSQGrad', 'VGGrad', \
                        'IG', 'SGIG', 'SGSQIG', 'VGIG', 'DLift', 'DLiftShap', \
                        'ShapValSample', 'ShapVal', 'lime', 'LRP', 'GBP', 'GuiGradCAM', 'InputXGrad'):
                        partial(get_captum_saliency, model, dataLoaderSal, saliency_id ,\
                                device, baseline = 'zero', layer=layer, pred_labels=pred_labels), 
     
     
                        ('GradAsc') : partial(get_grad_ascent_saliency, model, dataLoaderSal, iterations,\
                                              lr_sal, device, pred_labels=pred_labels), 
                        ('CaptumIG'): partial(get_captum_ig_saliency, model, dataLoaderSal, 0,\
                                              device, n_steps=steps, pred_labels=pred_labels),
    
                        ('LocalIGAll'): partial(get_local_ig_saliency, model, dataLoaderSal, baseline, \
                                                device, m_steps=steps, pred_labels=pred_labels), 
                        ("MultiGradAsc"): partial(get_multi_grad_ascent_max_saliency, model, dataLoaderSal, 
                                                  baseline, iterations, lr_sal, device, m_steps=50, pred_labels=pred_labels), 
                        ("WtPerturbation"): partial(saliency_through_weight_perturbation, model, 
                                                    model_dict, dataLoaderSal, iterations, device, pred_labels=pred_labels) 
    }
    
    for key in saliency_options.keys():
        if method in key:

            print('Working on {} method'.format(method))

            output_details = {0: method, 1: 'path_grad', 2: 'only_ig', 3: 'only_max'}
            output = saliency_options[key]()
            
            print("Output length:", len(output))
            
            break
                
    return output

def plot_maps_method_horizontal(
        model, 
        FinalData, 
        Labels, 
        classnames, 
        name,
        data_name, 
        saliency_dict, 
        method_captions, 
        p, 
        range_to_display = np.asarray(range(95, 105, 1)), 
        fig_size = (18, 18),
        cm=None, 
        vis_sign='positive'
    ):
    
    saliency_methods_total_id = len(saliency_dict)
    image_labels = Labels[range_to_display[:]]

    model.eval()
    with torch.no_grad():
        output = model(FinalData) # We don't need to unsqueeze(1) if we already have that dimension as in color image
    
    preds = output.data.max(1, keepdim=True)[1]
    print(preds.shape)
    print(torch.flatten(preds[range_to_display[:]]))
    
    fig, axes = plt.subplots(nrows = 10, ncols = saliency_methods_total_id+1, sharex = "all", figsize=fig_size, squeeze=False)


    for i in range(10):
        
        sample = saliency_dict[0][range_to_display[i]]
        attribution = np.transpose(sample, (1,2,0))
        
        img = FinalData[range_to_display[i]].numpy()
        
        if np.squeeze(img).ndim == 3:
            img = np.transpose(img, (1,2,0))
            
            
        if data_name == 'imgnet':
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
        elif data_name == 'cifar10':
            mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
            std=[x/255.0 for x in [63.0, 62.1, 66.7]]
        
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt_fig, plt_axis = viz.visualize_image_attr(attribution,
                                                     img,
                                                     "original_image",
                                                     "all",
                                                     (fig.number, axes[i,0]),
                                                     title="",
                                                     use_pyplot = False)
        
        pred_title = str(classnames[preds[range_to_display[i]].item()])
        axes[i, 0].set_ylabel("Pr: "+pred_title, fontsize='x-small')

    for method_id in range(saliency_methods_total_id):

        saliency = saliency_dict[method_id]

        for i in range(10):

            sample = saliency_dict[method_id][range_to_display[i]]
            attribution = np.transpose(sample, (1,2,0))
            
            fig_tuple = fig.number, axes[i, method_id+1]
            
            
            plt_fig, plt_axis = viz.visualize_image_attr(attribution,
                                                         img,
                                                         "heat_map",
                                                         vis_sign,
                                                         fig_tuple,
                                                         cmap = cm,
                                                         title="",
                                                         use_pyplot = False)

    axes[0, 0].set_title('Image/Data', fontsize='small')


    for method_id in range(saliency_methods_total_id):
        axes[0, method_id+1].set_title(method_captions[method_id], fontsize='small')

    plt.show()
    
    dis_range = str(range_to_display[0])+'_'+str(range_to_display[-1])
    path = os.path.join('./Plots/Real/', "method_research_"+name+'_'+data_name+'_h_view_'+vis_sign+'_'+dis_range+'_'+str(p))
    
    print(path)
    fig.savefig(path+'.svg', transparent=True, bbox_inches='tight', pad_inches=0)
    fig.savefig(path+'.pdf', format='pdf', dpi=300)
    fig.savefig(path+'.png', format='png', dpi=300)
    print('Plots Saved...', path)
    plt.close(fig)


def plot_maps_method_vertical(
        model, 
        FinalData, 
        Labels, 
        classnames, 
        name,
        data_name, 
        saliency_dict, 
        method_captions, 
        p, 
        range_to_display = np.asarray(range(95, 105, 1)), 
        fig_size = (18, 18),
        cm=None, 
        vis_sign='positive',
        save = False,
        fname = "my_file"
    ):
    saliency_methods_total_id = len(saliency_dict)
    image_labels = Labels[range_to_display[:]]

    model.eval()
    with torch.no_grad():
        output = model(FinalData) # We don't need to unsqueeze(1) if we already have that dimension as in color image
    
    preds = output.data.max(1, keepdim=True)[1]
    print(preds.shape)
    print(torch.flatten(preds[range_to_display[:]]))
    
    fig, axes = plt.subplots(nrows = saliency_methods_total_id+1, ncols = 10, sharex = "all", figsize=fig_size, squeeze=False)


    for i in range(10):
        
        sample = saliency_dict[0][range_to_display[i]]
        attribution = np.transpose(sample, (1,2,0))
        
        img = FinalData[range_to_display[i]].numpy()
        
        if np.squeeze(img).ndim == 3:
            img = np.transpose(img, (1,2,0))
            
        if data_name == 'imgnet':
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
        elif data_name == 'cifar10':
            mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
            std=[x/255.0 for x in [63.0, 62.1, 66.7]]
            
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt_fig, plt_axis = viz.visualize_image_attr(attribution,
                                                     img,
                                                     "original_image",
                                                     "all",
                                                     (fig.number, axes[0,i]),
                                                     title="",
                                                     use_pyplot = False)
        
#         pred_title = str(classnames[preds[range_to_display[i]].item()])
#         axes[0, i].set_title("Pr: "+pred_title, fontsize='medium')

    for method_id in range(saliency_methods_total_id):

        saliency = saliency_dict[method_id]

        for i in range(10):

            sample = saliency_dict[method_id][range_to_display[i]]
            attribution = np.transpose(sample, (1,2,0))
            
            fig_tuple = fig.number, axes[method_id + 1, i]
            
            
            plt_fig, plt_axis = viz.visualize_image_attr(attribution,
                                                         img,
                                                         "heat_map",
                                                         vis_sign,
                                                         fig_tuple,
                                                         cmap = cm,
                                                         title="",
                                                         use_pyplot = False)

#     axes[0, 0].set_title('Image/Data', fontsize='medium')


    for method_id in range(saliency_methods_total_id):
        axes[method_id + 1, 0].set_ylabel(method_captions[method_id], fontsize='medium')

    plt.show()
    
    dis_range = str(range_to_display[0])+'_'+str(range_to_display[-1])
    
    if save:
        path = os.path.join('./Plots/Real/', fname+'_'+cm+'_'+str(p) )
        print(path)
        fig.savefig(path+'.svg', transparent=True, bbox_inches='tight', pad_inches=0)
        fig.savefig(path+'.pdf', format='pdf', dpi=300)
        fig.savefig(path+'.png', format='png', dpi=300)
        print('Plots Saved...', path)
    plt.close(fig)


# for captum viz
import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap

def plot_model_randomization_maps(
        model, 
        FinalData, 
        Labels, 
        classnames, 
        name,
        data_name, 
        saliency_dict, 
        method_captions, 
        sample_no,
        p, 
        cm=None, 
        vis_sign='positive'
    ):
    
    """
    This method plots cacaded randomization test results
    """
    
    # 'keys' contains the layer's information
    layers = list(saliency_dict.keys())
    
    no_of_layers = len(layers)
    
    # how many methods were used for post hoc analysis
    id = len(saliency_dict[layers[0]])
   
    L = Labels[sample_no]

    model.eval()
    with torch.no_grad():
        output = model(FinalData) # We don't need to unsqueeze(1) if we already have that dimension as in color image
    
    preds = output.data.max(1, keepdim=True)[1]
    print(preds.shape)
    print(torch.flatten(preds[sample_no]))
    
    s = classnames[preds[sample_no].item()]
    pred_title = "_".join(s.split())
    
    
    img = FinalData[sample_no].numpy()
        
    if np.squeeze(img).ndim == 3:
        img = np.transpose(img, (1,2,0))
        
    if data_name == 'imgnet':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
            
    elif data_name == 'cifar10':
        mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
        std=[x/255.0 for x in [63.0, 62.1, 66.7]]
            
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    fig, axes = plt.subplots(nrows = id, ncols = no_of_layers, figsize=(15,10))


    for count, (layer_name, all_method_saliency) in enumerate(saliency_dict.items()):
        
#         attribution = np.transpose(sample, (1,2,0))
        
#         axes[i, 0].set_ylabel("Pr: "+pred_title, fontsize='medium')


        for method_id in range(id):

            saliency = all_method_saliency[method_id]
            sample = all_method_saliency[method_id][sample_no]
            attribution = np.transpose(sample, (1,2,0))

            fig_tuple = fig.number, axes[method_id, count]
            
            
            
            if np.all(attribution==0):
                attribution += 1e-5
#                 print(f'Layer Name:{layer_name}\nMethod Name: {method_captions[method_id]}')
            
            plt_fig, plt_axis = viz.visualize_image_attr(attribution,
                                                         img,
                                                         "heat_map",
                                                         vis_sign,
                                                         fig_tuple,
                                                         cmap = cm,
                                                         title="",
                                                         use_pyplot = False)
    
        axes[0, count].set_title(layers[count], fontsize='x-small')

#         axes[0, 0].set_title('Image/Data', fontsize='medium')


        for method_id in range(id):
            axes[method_id, 0].set_ylabel(method_captions[method_id], fontsize='x-small')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.show()

    path = os.path.join('./Plots/Real/', "method_research_"+name+'_'+data_name+'_'+pred_title+'_'+vis_sign+'_'+str(p) )
    print(path)
    fig.savefig(path+'.svg', transparent=True, bbox_inches='tight', pad_inches=0)
    fig.savefig(path+'.pdf', format='pdf', dpi=300)
    fig.savefig(path+'.png', format='png', dpi=300)
    print('Plots Saved...', path)
    plt.close(fig)
 
    