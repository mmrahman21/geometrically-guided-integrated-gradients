
import numpy as np
import scipy.stats
import glob
import sys

from skimage.metrics import structural_similarity as ssim
import torch

def mse(imageA, imageB):
    
    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse_error /= float(imageA.shape[0] * imageA.shape[1])
    
    return mse_error

def structural_sim(imageA, imageB):
    s = ssim(imageA, imageB)
    d = mse(imageA, imageB)
    
    return s, d

def robyn_method(Mask,Data):
    X = Mask*Data
    Y = Data
    Xs = np.sum(X)
    Ys = np.sum(Y)
    return Xs/Ys


def spear_rank(M,N):
    M = M.flatten()
    N = N.flatten()

    Ma = np.flip(np.argsort(M))
    Na = np.flip(np.argsort(N))

    Mv = np.copy(M)
    Nv = np.copy(N)

    count = M.shape[0]
    diff = 0
    for i in range(int(M.shape[0]/8)):
        Mv[Ma[i]] = count
        Nv[Na[i]] = count
        count -= 1
    for i in range(M.shape[0]):
        diff += (Mv[i] - Nv[i])**2
    diff = 6*diff/(M.shape[0]**3 - M.shape[0])

    return 1-diff


def w_jacc(M,N):
    mins = 0
    maxs = 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            mins += min(M[i,j],N[i,j])
            maxs += max(M[i,j],N[i,j])
    return mins/maxs

def corrs(M,N):
    X = []
    for i in range(M.shape[0]):
        s = np.corrcoef(M[i],N[i])
        X.append(np.nan_to_num(s[0,1]))
    return X
def relu(x):
    x[x<0] = 0.
    return x

def ReLU(x):
    return x * (x > 0)

def make_mask(tsize, csize, label):
    mask = np.zeros([tsize, csize])
    if label:
        mask[:,0:int(csize/2)] = 1
    else:
        mask[:,int(csize/2):] = 1
    return mask

def calcAllMetric(grads, FinalData, Labels, POS, method_name):
        
    out_0 = []
    out_1 = []
    
    print("Full Data Shape:", FinalData.shape)
    print("Position Shape:", POS.shape)

    for i in range(grads.shape[0]):
        data_sample = FinalData[i]
        g = grads[i]
        g = g / np.max(g)
        start = POS[i]

#         print(start[0], "and ", start[1])

        temp_mask_discrete = np.zeros(data_sample.shape)
        temp_mask = np.zeros(data_sample.shape)

        label = Labels[i]
        features = temp_mask_discrete.shape[0]
        
        # Use these setting for top/down box dataset. 

#         temp_mask_discrete[:, start:start+10] = 1   # for the old two class data
#         temp_mask[:, start:start+10]= data_sample[:, start:start+10] # OR ReLU ? It is using data to create mask.
       
#         temp_mask = ReLU(temp_mask)

#         if label:
#             temp_mask_discrete[int(features / 2):, :] = 0
#             temp_mask[int(features / 2):, :] = 0
#         else:
#             temp_mask_discrete[0:int(features / 2), :] = 0
#             temp_mask[int(features / 2):, :] = 0
            
#         temp_mask = temp_mask / np.max(temp_mask)  # if data used as a mask

         # for the actual spatial data 
    
        temp_mask_discrete[start[1]:start[1]+30, start[0]:start[0]+10] = 1  
        
        temp_mask = np.copy(temp_mask_discrete)

        # For the three class data ...
    
#         if label == 0:
#             temp_mask_discrete[0:int(features/3), start:start+10] = 1
#         elif label == 1:
#             temp_mask_discrete[int(features/3):int(features*2/3), start:start+10] = 1
#         else:
#             temp_mask_discrete[int(features*2/3):, start:start+10] = 1
            
#         print(start) 
        
        # for the multi way two class data 
        
#         for k in range(3):
#             temp_mask_discrete[start[k,1]:start[k,1]+30, start[k,0]:start[k,0]+10]=1

        print(np.sum(temp_mask_discrete))
        print(np.sum(temp_mask))

        T1 = scipy.stats.pearsonr(g.flatten(), temp_mask.flatten())
        T2 = scipy.stats.spearmanr(g.flatten(),temp_mask.flatten(), nan_policy="omit")
        
        pearson_r = T1[0]
        spearman_r = T2[0] 

        wj = w_jacc(g,temp_mask)
        rm = robyn_method(temp_mask_discrete, g)
        
        s_sim, mean_sqErr = structural_sim(temp_mask_discrete, g)
        
        print(f"Method: {method_name}\nPearson Correlation: {pearson_r}\nSpearman Correlation: {spearman_r}\nWeighted Jaccard: {wj}\nRobyn Method:{rm}\nSSIM: {s_sim}\nMSE: {mean_sqErr}")
        if label == 0:
            out_0.append(np.array([pearson_r, spearman_r, wj, rm, s_sim, mean_sqErr]))
        else:
            out_1.append(np.array([pearson_r, spearman_r, wj, rm, s_sim, mean_sqErr]))
        
    return out_0, out_1


def calc_metric_for_real_data(grads, FinalData, Labels, total_labels, method_name):
        
    out = {i: [] for i in range(total_labels)}
    
    print("Full Data Shape:", FinalData.shape)
    print("Gradient Shape:", grads.shape)
    
    grads = np.squeeze(grads)
    FinalData = torch.squeeze(FinalData)
    FinalData = FinalData.numpy()
    
    print("Full Data Shape:", FinalData.shape)
    print("Gradient Shape:", grads.shape)
    print('# Using new data scaling technique.....')

    for i in range(grads.shape[0]):
       
        # Previously used
        
#         data = np.abs(FinalData[i])
#         data = data / np.max(data)
        
        # Now using
        data = FinalData[i]
        data = np.clip(data, -1, 1)
        data = (data + 1.0)/2.0
        
#         mask = 1.0*(FinalData[i] > 0)
        mask = data
        
        g = grads[i]
        g = g / np.max(g)

        label = Labels[i]

        T1 = scipy.stats.pearsonr(g.flatten(), mask.flatten())
        T2 = scipy.stats.spearmanr(g.flatten(),mask.flatten(), nan_policy="omit")
        
        pearson_r = T1[0]
        spearman_r = T2[0] 

        wj = w_jacc(g,mask)
        rm = robyn_method(mask, g)
        
        s_sim, mean_sqErr = structural_sim(mask, g)
        
        print(f"Method: {method_name}\nPearson Correlation: {pearson_r}\nSpearman Correlation: {spearman_r}\nWeighted Jaccard: {wj}\nRobyn Method:{rm}\nSSIM: {s_sim}\nMSE: {mean_sqErr}")
        
        out[label.item()].append(np.array([pearson_r, spearman_r, wj, rm, s_sim, mean_sqErr]))
        
    return out