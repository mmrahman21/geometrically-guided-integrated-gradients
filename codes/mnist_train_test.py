import matplotlib.pylab as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim 
from models import VGGNet, LeNet, MNNet, MNNet2, SanityCheckCNN
from dnn_utils import update_lr, schedule_lr_decay
from datetime import datetime
import sys
import os
import time
import gc


n_epochs = 100 # 3000
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.0004
momentum = 0.9
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model_id = int(sys.argv[1])
print(f"Model ID: {model_id}")

trainX, trainY = torch.load('./my_mnist/MNIST/processed/training.pt')
print(trainX.shape)
testX, testY = torch.load('./my_mnist/MNIST/processed/test.pt')
print(testX.shape)


trainX = trainX.unsqueeze(1).float()
trainX /= 255
trainY = trainY.long()

# Shuffle training labels
id = torch.randperm(trainY.shape[0])
shuffled_trainY = trainY[id].view(trainY.size())

print(trainX.shape)
testX = testX.unsqueeze(1).float()
testX /= 255
testY = testY.long()
print(testX.shape)

transform=torchvision.transforms.Compose([
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])

trainX = transform(trainX)
testX = transform(testX)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(trainX, trainY), batch_size=batch_size_train, shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(testX, testY), batch_size=batch_size_test, shuffle=True)


def train(epoch):
    model.train()
    correct_train = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct_train += pred.eq(target.data.view_as(pred)).sum()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(model.state_dict(), './models and saliencies/method_research_mnist_model_'+str(model_id)+'.pth')
            torch.save(optimizer.state_dict(), './models and saliencies/method_research_mnist_optim_'+str(model_id)+'.pth')
    print('\nTrain set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct_train, len(train_loader.dataset),
    100. * correct_train / len(train_loader.dataset)))
            
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    return test_loss

model = SanityCheckCNN()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)

# network_state_dict = torch.load('./models and saliencies/method_research_mnist_model_'+str(model_id)+'.pth')
# model.load_state_dict(network_state_dict)

# optimizer_state_dict = torch.load('./models and saliencies/method_research_mnist_optim_'+str(model_id)+'.pth')
# optimizer.load_state_dict(optimizer_state_dict)

# optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.001) # fine tuned the lr
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

start_time = time.time()

test(model)
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test_loss = test(model)
#     scheduler.step(test_loss)
    
elapsed_time = time.time() - start_time
print('Total Time Elapsed:', elapsed_time)
print(datetime.now())
