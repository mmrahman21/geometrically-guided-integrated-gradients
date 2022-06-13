import numpy as np
import torch
import sys
import os
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import defaultdict

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


ensembles = {0:'', 1:'smoothgrad', 2:'smoothgrad_sq', 3: 'vargrad', 4:'', 5:'smoothgrad', 6:'smoothgrad_sq', 7: 'vargrad'}

saliency_options = {0:'Grad', 1:'SGGrad', 2: 'SGSQGrad', 3: 'VGGrad', 4: 'IG', 5: 'SGIG', 6:'SGSQIG', 7:'VGIG', 8: 'DLift', 9: 'DLiftShap', 10: 'ShapValSample', 11: 'ShapVal', 12: 'lime'}

def compute_global_vec(model, x, device):
        sx = []
        for episode in x:
            mean = episode.mean()
            sd = episode.std()
            episode = (episode - mean) / sd
            sx.append(episode)

        x = torch.stack(sx)
        
        x = x.permute(0, 2, 1)
        
        b_size = x.size(0)
        s_size = x.size(1)
        

        model.lstm_hidden = model.init_hidden(b_size, device)
        lstm_out, model.lstm_hidden = model.lstm(x, model.lstm_hidden)
        
        outputs = lstm_out
      
        # For anchor point
        B= outputs[:,-1, :]
        B = B.unsqueeze(1).expand_as(outputs)
        outputs2 = torch.cat((outputs, B), dim=2)
        
        
        # For attention calculation
        b_size = outputs2.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs2.reshape(-1, 2*model.hidden)

        weights = model.attn(out)
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)

        # Batch-wise multiplication of weights and lstm outputs

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        attn_applied = attn_applied.squeeze()
        attn_applied = attn_applied.view(b_size, -1)

        return attn_applied
    
def get_layer_to_input_saliency(model, loaderSal, iterations, lr, device):
    
#     model.eval()    
# m = img.mean()
# s = img.std()
# fn = torchvision.transforms.Normalize(mean=[m], std=[s])
# img = fn(img)

    all_saliencies = defaultdict(list)
    
    for i, data in enumerate(loaderSal):
        if i % 50 == 0:
            print("Processing subject: {}".format(i))
        x, y = data   
        
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_()
        
        saliencies = []
        
        for j in range(iterations):
            
            out = compute_global_vec(model, x, device)

            loss = out.norm()
            if i % 50 == 0:
                print(out.shape)
                print(loss)
                
            model.zero_grad()
            loss.backward()
            
            # Normalize the gradients.
            # gradients /= tf.math.reduce_std(gradients) + 1e-8 
            
            x.grad.data = x.grad.data / (torch.std(x.grad.data) + 1e-8)
            
            x.data = x.data + lr * x.grad.data
    
            saliencies.append(np.squeeze(x.cpu().detach().numpy())) # saving grad for checking instead of accumulation
            x.grad.zero_()
    
        saliencies = np.stack(saliencies, axis=0)
        print(saliencies.shape)
        all_saliencies[i] = saliencies
    
    all_saliency_result = list(all_saliencies.values())
    all_saliency_result = np.array(all_saliency_result)
    print("All Saliencies Combined:", all_saliency_result.shape)
    
    return all_saliency_result

# def get_layer_to_input_saliency(model, loaderSal, iterations, lr, device):
    
#     all_saliencies = defaultdict(list)
    
#     saliencies = []
#     for i, data in enumerate(loaderSal):
#         if i % 50 == 0:
#             print("Processing subject: {}".format(i))
#         x, y = data   
        
#         x = x.to(device)
#         y = y.to(device)
#         x.requires_grad_()
        
#         for j in range(iterations):
            
#             out = compute_global_vec(model, x, device)

#             loss = out.norm()
#             if i % 50 == 0:
#                 print(out.shape)
#                 print(loss)
                
#             model.zero_grad()
#             loss.backward()
#             x.data = x.data + lr * x.grad.data
    
#         saliencies.append(np.squeeze(x.cpu().detach().numpy()))
    
#     saliencies = np.stack(saliencies, axis=0)
#     print(saliencies.shape)
             
#     return saliencies

def get_nst_like_saliency(model, loaderSal, start_with, iterations, learning_rate, device):
    
    print("NST like saliency computation...")
    saliencies = []
    for i, data in enumerate(loaderSal):
        if i % 50 == 0:
            print("Processing subject: {}".format(i))
        x, y = data   
        
        if start_with == "white":
            white_noise_input = np.random.uniform(-1., 1., x.shape).astype(np.float32)
#         else:
#             gaussian_noise_input = np.random.normal(loc=0, scale=1., size=x.shape).astype(np.float32)
          
        init_img = torch.from_numpy(white_noise_input).float().to(device)
    
        optimizing_img = Variable(init_img, requires_grad=True)
        
        x = x.to(device)
        y = y.to(device)
        optimizer = optim.Adam((optimizing_img,), lr=learning_rate)
        
        for j in range(iterations):
            
            x_out = compute_global_vec(model, x, device)
            opt_out = compute_global_vec(model, optimizing_img, device)
            
            diff = x_out - opt_out

            loss = diff.norm()*(1/len(diff))
            
            if i % 50 == 0:
                print(diff.shape)
                print(loss)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
    
        saliencies.append(np.squeeze(optimizing_img.cpu().detach().numpy()))
    
    saliencies = np.stack(saliencies, axis=0)
    print(saliencies.shape)
             
    return saliencies

def get_logit_to_input_saliency(model, loaderSal, iterations, lr, device):
    
#     model.eval()
    all_saliencies = defaultdict(list)
    
    for i, data in enumerate(loaderSal):
        if i % 50 == 0:
            print("Processing subject: {}".format(i))
        x, y = data   
        
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_()
        
        saliencies = []
        
        for j in range(iterations):
            
            out = model(x)
#             loss = out[:, y]
            grad_outputs = torch.zeros(x.shape[0], 2).to(device)
            grad_outputs[:, y] = 1
            
            if i % 50 == 0:
                print(f"Logit Score: {out[:, y]}")
                
            model.zero_grad()
#             loss.backward()
            out.backward(gradient=grad_outputs)
    
            # Normalize the gradients.
            # gradients /= tf.math.reduce_std(gradients) + 1e-8 
            
            x.grad.data = x.grad.data / (torch.std(x.grad.data) + 1e-8)
            
            x.data = x.data + lr * x.grad.data
            
            saliencies.append(np.squeeze(x.cpu().detach().numpy()))  # saving grad for checking instead of accumulation
            x.grad.zero_()
    
        saliencies = np.stack(saliencies, axis=0)
        print(saliencies.shape)
        all_saliencies[i] = saliencies
    
    all_saliency_result = list(all_saliencies.values())
    all_saliency_result = np.array(all_saliency_result)
    print("All Saliencies Combined:", all_saliency_result.shape)
    
    return all_saliency_result

# def get_logit_to_input_saliency(model, loaderSal, iterations, lr, device):
    
#     all_saliencies = defaultdict(list)
#     saliencies = []
#     for i, data in enumerate(loaderSal):
#         if i % 50 == 0:
#             print("Processing subject: {}".format(i))
#         x, y = data   
        
#         x = x.to(device)
#         y = y.to(device)
#         x.requires_grad_()
        
#         for j in range(iterations):
            
#             out = model(x)
#             loss = out[:, y]
            
#             if i % 50 == 0:
#                 print(loss)
                
#             model.zero_grad()
#             loss.backward()
#             x.data = x.data + lr * x.grad.data
    
#         saliencies.append(np.squeeze(x.cpu().detach().numpy()))
    
#     saliencies = np.stack(saliencies, axis=0)
#     print(saliencies.shape)
             
#     return saliencies
    

def get_captum_saliency(model, loaderSal, saliency_id, device, baseline ='zero', layer=None, pred_labels=None):
    
    '''Takes a trained model, the loader to load saliency data, and saliency_id referring to saliency method, 
    computes and returns saliency for all the data'''
    
    print(next(model.parameters()).is_cuda)
    
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    if saliency_id <4:
        sal = Saliency(model)
        
    elif saliency_id <=7:
        sal = IntegratedGradients(model)
    
    elif saliency_id == 8:
        print('Multiplication False')
        dl = DeepLift(model, False)
        
    elif saliency_id == 9:
        dlshap = DeepLiftShap(model)
        
    elif saliency_id == 10:
        svs = ShapleyValueSampling(model)
            
    elif saliency_id == 11:
        # all permutations 
        sv = ShapleyValues(model)
        
    elif saliency_id == 12:
        lime = Lime(model)
        
    elif saliency_id == 22:
        lrp = LRP(model)
        
    elif saliency_id == 23:
        gbp = GuidedBackprop(model)
        
    elif saliency_id == 24:
        guided_gc = GuidedGradCam(model, layer)
        
    elif saliency_id == 25:
        input_x_gradient = InputXGradient(model)
            
    else:
        print('Wrong Method')
        return None
        
    if saliency_id !=0 and saliency_id !=4 and saliency_id <=7:
        nt = NoiseTunnel(sal)
        print('Ensemble Begins .. with', saliency_options[saliency_id])
        print('Ensembles:', ensembles[saliency_id])
    
    saliencies = []
    
#     x_sample, y_sample = next(iter(loaderSal))
        
    for i, data in enumerate(loaderSal):
        if i % 1000 == 0:
            print(i)
        x, y = data 
        
        x = x.to(device)
        y = y.to(device)
        
        print(x.shape)
        
        if not pred_labels:
            outputs = model(x)
            _, preds = torch.max(outputs.data, 1)
            # print('True ={}, Predicted= {}'.format(y.item(), preds.item()))
        else:
      
            preds = pred_labels[i]
        
        if baseline == 'zero':
            bl = torch.zeros(x.shape).to(device)
            if i == 0:
                print('Using zero baseline...')
                
        elif baseline == 'random':
            bl = torch.normal(0, 1, size=x.shape).to(device)
            if i == 0:
                print('Using random baseline...')

        else:
            print('No baseline is needed for this method...')
        
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_()
        
        if saliency_id == 0:
            print('Computing:', saliency_options[saliency_id])
            S = sal.attribute(x, target=preds, abs=False)
            
        elif saliency_id == 4:
            print('Computing:', saliency_options[saliency_id])
            S = sal.attribute(x,bl,target=preds)
            
        elif saliency_id > 0 and saliency_id < 4:
            S = nt.attribute(x, nt_type=ensembles[saliency_id], nt_samples=10, target=preds, abs=False)
            
        elif saliency_id <= 7:
            S = nt.attribute(x, nt_type=ensembles[saliency_id], nt_samples=10, baselines = bl, target=preds)
            
        elif saliency_id == 8:
            S = dl.attribute(x, bl, target=preds)
            
        elif saliency_id == 9:
            print('Shapley values based on Deep Lift')
            batch_size = x.shape[0]
            baseline_single=torch.from_numpy(np.random.random(x.shape)).to(device)                       
#             baseline_multiple=torch.from_numpy(np.random.random((x.shape[0]*5,x.shape[1],x.shape[2]))).to(device)
#             print(baseline_multiple.shape)
            S = dlshap.attribute(x, baselines=baseline_single, target=preds)
            
        elif saliency_id == 10:
            # input size is huge and seems computation time is impractical.
            # We can use feature mask for this, either in time steps or component wise.
            S = svs.attribute(x, target=preds, n_samples=100)
            
        elif saliency_id == 11:
            # input size is huge and seems computation time is abosolutely impractical.
            S = sv.attribute(x, target=preds)
            
        elif saliency_id == 12:
            # We can use feature mask for this, either in time steps or component wise instead of n_samples
            S = lime.attribute(x, target=preds, n_samples=20)
            
        elif saliency_id == 22:
            S = lrp.attribute(x, target=preds)
            
        elif saliency_id == 23:
            S = gbp.attribute(x, target=preds)
        
        elif saliency_id == 24:
            S = guided_gc.attribute(x, target=preds)
            
        elif saliency_id == 25:
            S = input_x_gradient.attribute(x, target=preds)

        saliencies.append(np.squeeze(S.cpu().detach().numpy()))
    
    saliencies1 = np.stack(saliencies, axis=0)
    print(saliencies1.shape)
             
    return saliencies1


def get_captum_saliency_norm_data(model, loaderSal, saliency_id, device):
    
    '''Takes a trained model, the loader to load saliency data, and saliency_id referring to saliency method, 
    computes and returns saliency for all the data'''
    
    print("Saliency on z-score")
    
    model.zero_grad()
    
    if saliency_id <4:
        sal = Saliency(model)
        
    elif saliency_id <=7:
        sal = IntegratedGradients(model)
    
    elif saliency_id == 8:
        dl = DeepLift(model)
        
    elif saliency_id == 9:
        dlshap = DeepLiftShap(model)
        
    elif saliency_id == 10:
        svs = ShapleyValueSampling(model)
            
    elif saliency_id == 11:
        # all permutations 
        sv = ShapleyValues(model)
        
    elif saliency_id == 12:
        lime = Lime(model)
        
    else:
        print('Wrong Method')
        return None
        
#     sal = GuidedBackprop(model)

    if saliency_id !=0 and saliency_id !=4 and saliency_id <=7:
        nt = NoiseTunnel(sal)
        print('Ensemble Begins .. with', saliency_options[saliency_id])
        print('Ensembles:', ensembles[saliency_id])
    
    saliencies = []
    for i, data in enumerate(loaderSal):
        if i % 1000 == 0:
            print(i)
        x, y = data  
        
        mean = x.mean()
        sd = x.std()
        x = (x - mean) / sd
        
        outputs = model(x)
        _, preds = torch.max(outputs.data, 1)
#         print('True ={}, Predicted= {}'.format(y.item(), preds.item()))
        
        bl = torch.zeros(x.shape).to(device)
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_()
        
        if saliency_id == 0:
            S = sal.attribute(x, target=preds, abs=False)
            
        elif saliency_id == 4:
            S = sal.attribute(x,bl,target=preds)
            
        elif saliency_id > 0 and saliency_id < 4:
            S = nt.attribute(x, nt_type=ensembles[saliency_id], nt_samples=10, target=preds, abs=False)
            
        elif saliency_id <= 7:
            S = nt.attribute(x, nt_type=ensembles[saliency_id], nt_samples=10, baselines = bl, target=preds)
            
        elif saliency_id == 8:
#             print('Deep Lift')
            S = dl.attribute(x, target=preds)
            
        elif saliency_id == 9:
            print('Shapley values based on Deep Lift')
            batch_size = x.shape[0]
            baseline_single=torch.from_numpy(np.random.random(x.shape)).to(device)                       
            baseline_multiple=torch.from_numpy(np.random.random((x.shape[0]*5,x.shape[1],x.shape[2]))).to(device)
            print(baselin_multiple.shape)
            S = dlshap.attribute(x, baselines=baseline_multiple, target=preds)
            
        elif saliency_id == 10:
            # input size is huge and seems computation time is impractical.
            # We can use feature mask for this, either in time steps or component wise.
            S = svs.attribute(x, target=preds, n_samples=10)
            
        elif saliency_id == 11:
            # input size is huge and seems computation time is abosolutely impractical.
            S = sv.attribute(x, target=preds)
            
        elif saliency_id == 12:
            # We can use feature mask for this, either in time steps or component wise instead of n_samples
            S = lime.attribute(x, target=preds, n_samples=20)

        saliencies.append(np.squeeze(S.cpu().detach().numpy()))
    
    saliencies1 = np.stack(saliencies, axis=0)
    print(saliencies1.shape)
             
    return saliencies


def load_pretrain_model(model, exp_type, device):
    
    '''Takes a randomly initialized model and loads with pretrained weights, 
    if exp_type is 'UFPT', otherwise returns the original'''
    
    if exp_type == 'UFPT':
        
#         modelpath = '../wandb/SynDataPretrainedEncoder/InterpolationVARDataEncoder/HCP/HCP_LSTM_only_gain_0.1_Aug062020_1.pt'
#         modelpath = '../wandb/SynDataPretrainedEncoder/InterpolationVARDataEncoder/HCP/HCP_LSTM_only_gain_0.1_Nov302020_1.pt'
#         modelpath = '../wandb/SynDataPretrainedEncoder/InterpolationVARDataEncoder/HCP/HCP_LSTM_only_gain_0.1_April062021_Anchor_Encoder_Attn_1.pt'
#         modelpath = '../wandb/SynDataPretrainedEncoder/InterpolationVARDataEncoder/HCP/HCP_LSTM_only_gain_0.1_April062021_Bidirected_Encoder_1.pt'

        modelpath = '../wandb/SynDataPretrainedEncoder/InterpolationVARDataEncoder/HCP/HCP_LSTM_only_gain_0.1_May032021_Encoder_Anchoring_1.pt'
   
        modelA_dict = torch.load(modelpath, map_location=device)  # Pre-trained model is model A

        print('Model loaded from here:', modelpath)

        modelB_dict = model.state_dict()    # Let's assume downstream model as B

        print("modelB (downstream model) is going to use the common layers parameters from modelA")
        pretrained_dict = modelA_dict
        model_dict = modelB_dict

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        model.load_state_dict(model_dict)
        model.to(device)

        print('PRE-TRAINED MODELS LOADED')
        return model 
    
    else:
        print('NO MODEL LOADING IS NEEDED , IT WILL BE TRAINED AFRESH...')
        model.to(device)
        return model
    

def save_predictions(X, model, dataloader, prediction_path):
    
    '''Takes a trained model, dataloader and a prediction path, saves in prediction path'''

    # Only for saving predictions and true labels
    predictions = np.zeros((X.shape[0], 2))
    for i, d in enumerate(dataloader):
        if i % 1000 == 0:
            print(i)
        x, y = d  
        
        outputs = model(x)
        _, preds = torch.max(outputs.data, 1) 
        print(preds)
        accuracy = (preds == y).sum().item()
        predictions[:, 0] = y.cpu().numpy()
        predictions[:, 1] = preds.cpu().detach().numpy()
        print('Acc obtained overall:', accuracy)
    
    np.save(prediction_path, predictions)
    print('Saving here...', prediction_path)
    
    return outputs

def save_reload_model(model, model_path, prefix, device, restart, save_reload = 'save'):
    
    '''Saves or reloads model to a specified basepath, and file prefix'''
    
    model_path = os.path.join(model_path, prefix+'_captum_use_'+str(restart)+ '.pt')
    
    # Save model
    if save_reload == 'save':
        torch.save(model.state_dict(), model_path)
        print("Model saved here:", model_path)
        
    # Reload model 
    elif save_reload == 'reload':
        model_dict = torch.load(model_path, map_location=device)  # with good components
        model.load_state_dict(model_dict)
        model.to(device)
        print('Model loaded from:', model_path)
        return model
        
def save_acc_auc(model_path, prefix, result):
    
    dframe = pd.DataFrame(result, columns=["ACC", "AUC"])
    resultfname = os.path.join(model_path, prefix +'_acc_auc.csv')
    dframe.to_csv(resultfname)
    print('Accuracy and area under curve saved successfully..')
    
    
   


   

