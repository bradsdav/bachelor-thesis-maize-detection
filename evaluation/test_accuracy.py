import os
import numpy as np
from src import dataset
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Bottleneck, ResNet, ResNet50

print("ok")

batch_size = 200

testDataset = dataset.myDataset('./data/rand/test.pkl', './data/rand/test_labels.pkl')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net_b = ResNet50(2)
net_b.load_state_dict(torch.load('./weights/weights_dr.pts', map_location='cpu'))
net_b = net_b.to(device)

net_r = ResNet50(2)
net_r.load_state_dict(torch.load('./weights/weights_hid_dr.pts', map_location='cpu'))
net_r = net_r.to(device)

criterion = nn.BCELoss()

testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=6)

print("Testing both networks!")

accuracy_base= 0
accuracy_rob = 0
test_len = 0
    
for i, data in enumerate(testloader):
    input = data['data'].to(device)
    label = data['label'].to(device)
    
    test_len += input.size(0)

    with torch.no_grad():
        output_b = net_b(input)

    for k in range(len(output_b)):
        if output_b[k][0] > output_b[k][1]:
            if label[k][0] == 1:
                accuracy_base += 1
        else:
            if label[k][1] == 1:
                accuracy_base += 1

    with torch.no_grad():
        output_r = net_r(input)

    for k in range(len(output_r)):
        if output_r[k][0] > output_r[k][1]:
            if label[k][0] == 1:
                accuracy_rob += 1
        else:
            if label[k][1] == 1:
                accuracy_rob += 1
    
    

print('Accuracy of baseline model on complete data:', 100*accuracy_base/test_len)
print('Accuracy of robust model on complete data:', 100*accuracy_rob/test_len)
print('')









accuracy_base= 0
accuracy_rob = 0
test_len = 0
    
for i, data in enumerate(testloader):
    input = data['data'].to(device)
    label = data['label'].to(device)

    hid = random.randint(0, 73)
    mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
    input = input * mask
    
    test_len += input.size(0)

    with torch.no_grad():
        output_b = net_b(input)

    for k in range(len(output_b)):
        if output_b[k][0] > output_b[k][1]:
            if label[k][0] == 1:
                accuracy_base += 1
        else:
            if label[k][1] == 1:
                accuracy_base += 1

    with torch.no_grad():
        output_r = net_r(input)

    for k in range(len(output_r)):
        if output_r[k][0] > output_r[k][1]:
            if label[k][0] == 1:
                accuracy_rob += 1
        else:
            if label[k][1] == 1:
                accuracy_rob += 1
    


print('Accuracy of baseline model on shorter time series:', 100*accuracy_base/test_len)
print('Accuracy of robust model on shorter time series:', 100*accuracy_rob/test_len)
print('')
