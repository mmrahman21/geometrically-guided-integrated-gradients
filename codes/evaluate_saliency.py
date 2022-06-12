# Synthetic data visualization

import numpy as np
import pandas as pd
import os
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
import torch
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from source.utils import get_argparser
from myscripts.generateSynData import artificial_batching_patterned_space1, artificial_batching_patterned_space2, three_class, actual_spatial, multi_way_data, artificial_batching_trend, my_multi_block_var_data, my_uniform_top_down, generate_top_down_pos_neg, generate_top_down_neg_neg, generate_top_down_zero
from myscripts.stitchWindows import stitch_windows
import time
import sys
import copy
from methods.saliency_evaluation_methods import calcAllMetric

# %matplotlib inline

from myscripts.metricMeasureNew import calcMetric, newMetric
from interpretability import post_process_saliency


def compute_z(saliency):
    normalized_sal = (saliency - saliency.mean())/ saliency.std()
    return normalized_sal
    

start_time = time.time()

components = 50
time_points = 140
window_shift = 1
samples_per_subject = 121
sample_y = 20


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

def ReLU(x):
    return x * (x > 0)


saliency_options = {0:'Grad', 1:'SGGrad', 2: 'SGSQGrad', 3: 'VGGrad', 4: 'IG', 5: 'SGIG', 6:'SGSQIG', 7:'VGIG', 8: 'DLift', 9: 'DLiftShap', 10: 'ShapValSample', 11: 'ShapVal', 12: 'lime', 13: 'DDContext', 14: 'DDLogit'}

Dataset = {0: 'oldtwocls', 1: 'newtwocls', 2: 'threecls', 3: 'multiwaytwocls', 4: 'uniformtwocls', 5: 'topdownposneg', 6: 'topdownnegneg', 7: 'topdownzero'}

DatasetDict = {"oldtwocls": artificial_batching_patterned_space2, "newtwocls": actual_spatial, "threecls": three_class, "multiwaytwocls": multi_way_data, "uniformtwocls": my_uniform_top_down, "topdownposneg": generate_top_down_pos_neg, "topdownnegneg": generate_top_down_neg_neg, "topdownzero": generate_top_down_zero}

MainDirectories = { 'oldtwocls': '','newtwocls' : 'newTwoClass', 'threecls' : 'threeClass', 'multiwaytwocls':'multiwayTwoClass', 'uniformtwocls': '', 'topdownposneg': '', 'topdownnegneg': '', 'topdownzero': ''}

data = 1

# Data, labels, start_positions, masks = artificial_batching_patterned_space2(3000, 140, 50, seed=1988)
# Data, labels, start_positions, masks = my_uniform_top_down(3000, 140, 50, seed=1988)
# Data, labels, start_positions, masks = generate_top_down_pos_neg(3000, 140, 50, seed=1988)
# Data, labels, start_positions, masks = generate_top_down_neg_neg(3000, 140, 50, seed=1988)
# Data, labels, start_positions, masks = generate_top_down_zero(3000, 140, 50, seed=1988)

Data, labels, start_positions, masks = actual_spatial(3000, 140, 50, seed=1988)
# Data , labels, start_positions = artificial_batching_trend(A, B, C, 20000, 140, A.shape[0], p_steps=20, alpha=0, seed=1988)
# Data, labels, start_positions, masks=multi_way_data(3000, 140, 50, seed=1988)
# Data, labels, start_positions, masks = three_class(3000, 140, 50, seed=1988)


print(Data[0, 0, 0:20])
Data = np.moveaxis(Data, 1, 2)

print('Original Data Shape:', Data.shape)

# FinalData = compute_z(Data[2000:])
FinalData = Data[2000:2500]
Labels = labels[2000:2500]
start_positions = start_positions[2000:2500]
print('Test Data Shape:', Labels.shape)


Gain = {1:0.1, 2:0.2, 3:0.3, 4:0.4, 5:0.5, 6:0.6, 7:0.7, 8:0.8, 9:0.9, 10:1.0, 11:1.1, 12:1.2, 13:1.3, 14:1.4, 15:1.5, 16:1.6, 17:1.7, 18:1.8, 19:1.9, 20:2.0}  

g = 1.0 

dir = "NPT"
mainDir = "newTwoClass" #"MyStride1Dir"

prefix = "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+"_"

prefix_gd = "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_Grad_zero_'
prefix_gd_input = "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_InputXGrad_zero_'
prefix_ig = "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_IG_zero_'
prefix_capt_ig = "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_CaptumIG_zero_'
prefix_local_ig= "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_LocalIGAll_zero_'
prefix_only_ig= "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_LocalIGAll_zero_only_ig_'
prefix_only_max= "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_LocalIGAll_zero_only_max_'
prefix_path_grad= "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_LocalIGAll_zero_path_grad_'
prefix_sgig = "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_SGIG_zero_'
prefix_gradasc ="method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_GradAsc_iter_100_lr_0.1_zero_'
prefix_multigradasc = "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_MultiGradAsc_iter_100_lr_0.1_zero_'
prefix_wt = "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_WtPerturbation_iter_100_zero_'
prefix_gbp = "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_GBP_zero_'
prefix_dlift= "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+'_DLift_random_'



positions = start_positions.astype(int)
POS = positions

iterations = 200


titles = ["GD", "Inp X GD", "IG", "L.IG","ONLY.IG", "SGIG", "GDAsc.M", "GDAsc.IG", "M.GDAsc.M", "M.GDAsc.IG","Wt.P.M", "Wt.P.IG", "GBP", "DLift"]

filenamepath = "../wandb"
filenamepath = os.path.join(filenamepath, 'Sequence_Based_Models', mainDir, dir, 'Saliency')

filenames = [prefix_gd, prefix_gd_input, prefix_ig, prefix_local_ig, prefix_only_ig, prefix_sgig, prefix_gradasc, prefix_multigradasc, prefix_wt, prefix_gbp, prefix_dlift]

titles = ["GD", "ONLY.IG", "ONLY.M", "GDAsc.IG","GDAsc.M", "M.GDAsc.IG", "M.GDAsc.M","Wt.P.IG", "Wt.P.M"]

filenames = [prefix_gd, prefix_only_ig, prefix_only_max, prefix_gradasc, prefix_multigradasc, prefix_wt]



out_0_metric = {k : [] for k in titles}
out_1_metric = {k : [] for k in titles}

for p in range(10):
    
    saliency_dict = {}
   
    id = 0
    
    print("Processing of gradients files started...")
    for fname in filenames:
        fname = os.path.join(filenamepath, fname+str(p)+'.npy')
        all_gradients = np.load(fname)
        print(fname)
        
        all_gradients = np.nan_to_num(all_gradients)
        

        array_has_nan = np.isnan(all_gradients).any()
        input_has_nan = np.isnan(FinalData).any()
        print('Grads have NaNs:', array_has_nan)
        print('Inputs have NaNs:', input_has_nan)
       
#         all_gradients = ReLU(all_gradients)
        all_gradients = np.abs(all_gradients)

        print(all_gradients.shape)
        if "Grad" in fname and not 'XGrad' in fname and len(all_gradients.shape) == 3:
            saliency_dict[titles[id]] = all_gradients
            id += 1
            print(f'Current ID: {id}')

        elif len(all_gradients.shape) == 3:
            saliency_dict[titles[id]] = all_gradients
            id += 1
            print(f'Current ID: {id}')

        elif "MultiDir" in fname and len(all_gradients.shape) == 4:
            max_sal, ig_like_sal, avg_sal = post_process_saliency(FinalData, all_gradients, scaling=False)
            saliency_dict[titles[id]] = max_sal
            saliency_dict[titles[id + 1]] = ig_like_sal
            saliency_dict[titles[id + 2]] = avg_sal
            id += 3
            print(f'Current ID: {id}')
        elif len(all_gradients.shape) == 4:
            max_sal, ig_like_sal, avg_sal = post_process_saliency(FinalData, all_gradients, scaling=False)
            saliency_dict[titles[id]] = ig_like_sal
            saliency_dict[titles[id + 1]] = max_sal
            id += 2
            print(f'Current ID: {id}')
    
    print(len(saliency_dict))
    print(id)
    
    
    for k, v in saliency_dict.items():
        
        # Calculate the Robyn metric here ....

        out_0, out_1 = calcAllMetric(saliency_dict[k], FinalData, Labels, POS, k)

        #     out_0, out_1 = newMetric(avg_saliency1, FinalData, Labels, POS)

        out_0_metric[k].append(np.array(out_0))
        out_1_metric[k].append(np.array(out_1))
        
        print('Calculated Metric Shape:')
        print(np.array(out_0_metric[k]).shape)
        print(np.array(out_1_metric[k]).shape)
        
            

# Save the metrics 

# np.savez(filenamepath+"/"+prefix+"out_0_all_metric_ReLUed_map.npz", **out_0_metric)  # absolute data
# np.savez(filenamepath+"/"+prefix+"out_1_all_metric_ReLUed_map.npz", **out_1_metric)

# np.savez(filenamepath+"/"+prefix+"out_0_all_metric_ReLUed_map_ReLUed_data.npz", **out_0_metric)
# np.savez(filenamepath+"/"+prefix+"out_1_all_metric_ReLUed_map_ReLUed_data.npz", **out_1_metric)

# np.savez(filenamepath+"/"+prefix+"out_0_all_metric_ReLUed_map_all_same_mask.npz", **out_0_metric)
# np.savez(filenamepath+"/"+prefix+"out_1_all_metric_ReLUed_map_all_same_mask.npz", **out_1_metric)

np.savez(filenamepath+"/"+prefix+"out_0_all_metrics_Absolute_map_all_same_mask.npz", **out_0_metric)
np.savez(filenamepath+"/"+prefix+"out_1_all_metrics_Absolute_map_all_same_mask.npz", **out_1_metric)


for k in titles:
    print(f"Method: {k}")
    print(np.array(out_0_metric[k]).shape)
    print(np.mean(out_0_metric[k], axis=1))
    print(np.mean(np.mean(out_0_metric[k], axis=1), axis=0))

    print(np.array(out_1_metric[k]).shape)
    print(np.mean(out_1_metric[k], axis=1))
    print(np.mean(np.mean(out_1_metric[k], axis=1), axis=0))

elapsed_time = time.time() - start_time
print('Total Time Elapsed:', elapsed_time)
    