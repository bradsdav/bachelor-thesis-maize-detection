import csv
import os
import numpy as np
from src import dataset
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Bottleneck, ResNet, ResNet50
from src.model_dropout import Bottleneck_drop, ResNet_drop, ResNet50_drop

print("ok")

batch_size = 100


validationDataset = dataset.myDataset('./data/rand/validation.pkl', './data/rand/validation_labels.pkl')
testDataset = dataset.myDataset('./data/rand/test.pkl', './data/rand/test_labels.pkl')

valloader = torch.utils.data.DataLoader(validationDataset, batch_size=batch_size, shuffle=False, num_workers=6)
testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=6)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ResNet50(2)
net.load_state_dict(torch.load('./weights/weights_hid_dr.pts', map_location='cpu'))
net = net.to(device)

net_drop = ResNet50_drop(2)
net_drop.load_state_dict(torch.load('./weights/weights_hid_dr.pts', map_location='cpu'))
net_drop = net_drop.to(device)

criterion = nn.BCELoss()

print("generating train")

for i, data in enumerate(valloader):
    input = data['data'].to(device)
    label = data['label'].to(device)

    with torch.no_grad():
        for x in range(2):
            output = net(input)
            print(output[0])
            print(output[1])
            print(output[2])
            print(output[3])
            print(output[4])
        
        for x in range(2):
            output = net_drop(input)
            print(output[0])
            print(output[1])
            print(output[2])
            print(output[3])
            print(output[4])

    break

