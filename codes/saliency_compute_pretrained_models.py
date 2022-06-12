import matplotlib.pylab as plt
import numpy as np
import torch
import torchvision
import torchvision.models as models
import sys
from PIL import Image
from torchvision import transforms
from torchvision import datasets, models, transforms
import os
from saliency_utilities import compute_saliency_for_methods
import copy
from datetime import datetime
import time
from methods.method_research_utilities import compute_accuracy, accuracy, load_imagenet_saliency_data,load_bgc_imagenet_saliency_data, load_cifar10_saliency_data, load_imagenet_saliency_metric_eval_data, load_imgnet_val_data


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
print("Loading pre-trained", MODELS[model_id], "model")
model = pretrained_models[MODELS[model_id]](pretrained=True)

model.to(device)

print(name)

model.eval()

unnormalized_resized_images, data, labels, dataLoaderSal, classes = load_imgnet_val_data()

# data, labels, dataLoaderSal, classes = load_imagenet_saliency_metric_eval_data()

x_batch, y_batch = next(iter(dataLoaderSal))
print(x_batch.shape, y_batch.shape)
    
# Read the imagenet categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    
# Compute Saliency
since = time.time()

data = data.to(device)
labels = labels.to(device)

outputs = model(data)
_, preds = torch.max(outputs, 1)
print(preds.shape)

# for pred in preds:
#     pred = pred.item()
#     print(categories[pred])
    
acc = compute_accuracy(outputs, labels)
print('Overall Acc: {}'.format(acc))
topk_acc = accuracy(outputs, labels, topk=(1, 2, 3, 4, 5))
print('TopK Acc: {}'.format(topk_acc))


baseline = torch.zeros(1, 3, 224, 224).to(device)

if method == 'WtPerturbation':
    iterations = 200
else:
    iterations = 100
    
lr_sal = 0.0001
prefix = 'method_research_imgnet_valSet_metricEval_'+name
model_dict = copy.deepcopy(model.state_dict())
out = compute_saliency_for_methods(
        model,
        model_dict,
        dataLoaderSal,
        method,
        baseline,
        iterations, 
        lr_sal,
        prefix,
        layer = None,
        steps = 50,
        device = device, 
        seed = 0
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
    
elapsed_time = time.time() - since
print('Total Time Elapsed:', elapsed_time)
print(datetime.now())