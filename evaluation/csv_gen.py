import csv
import random
import os
import numpy as np
from src import dataset
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Bottleneck, ResNet, ResNet50

print("ok")

batch_size = 100


validationDataset = dataset.myDataset('./data/orig/validation_or.pkl', './data/orig/validation_labels_or.pkl')
testDataset = dataset.myDataset('./data/orig/test_or.pkl', './data/orig/test_labels_or.pkl')

valloader = torch.utils.data.DataLoader(validationDataset, batch_size=batch_size, shuffle=True, num_workers=6)
testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=6)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ResNet50(2)
net.load_state_dict(torch.load('./weights/weights_hid_or_dr.pts', map_location='cpu'))
net = net.to(device)

criterion = nn.BCELoss()
                

# print("generating val")

# with open('results/eval/val_rand_hid.csv', 'w', newline='') as csvfile:
#     valwriter = csv.writer(csvfile)
#     for i, data in enumerate(valloader):
#         input = data['data'].to(device)
#         label = data['label'].to(device)
#         # label = torch.reshape(label, (-1,))

#         hid = random.randint(0, 73)
#         mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
#         input = input * mask
#         perc = 100*(hid/73)

#         with torch.no_grad():
#             output = net(input)
#             to_write = [0,0,0,perc]
#             for k in range(len(output)):
#                 to_write[0] = output[k][0].item()
#                 to_write[1] = output[k][1].item()
#                 if label[k][0] == 1:
#                     to_write[2] = 0
#                 else:
#                     to_write[2] = 1
#                 valwriter.writerow(to_write)

print("generating val")                

with open('results/eval/val.csv', 'w', newline='') as csvfile:
    testwriter = csv.writer(csvfile)
    for i, data in enumerate(valloader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        # label = torch.reshape(label, (-1,))

        # hid = random.randint(0, 73)
        # mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        # input = input * mask
        # perc = 100*(hid/73)

        with torch.no_grad():
            output = net(input)
            to_write = [0,0,0]
            for k in range(len(output)):
                to_write[0] = output[k][0].item()
                to_write[1] = output[k][1].item()
                if label[k][0] == 1:
                    to_write[2] = 0
                else:
                    to_write[2] = 1
                testwriter.writerow(to_write)

print("generating val hid")                

with open('results/eval/val_rand_hid.csv', 'w', newline='') as csvfile:
    testwriter = csv.writer(csvfile)
    for i, data in enumerate(valloader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        # label = torch.reshape(label, (-1,))

        hid = random.randint(0, 73)
        mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        input = input * mask
        perc = 100*(hid/73)

        with torch.no_grad():
            output = net(input)
            to_write = [0,0,0,perc]
            for k in range(len(output)):
                to_write[0] = output[k][0].item()
                to_write[1] = output[k][1].item()
                if label[k][0] == 1:
                    to_write[2] = 0
                else:
                    to_write[2] = 1
                testwriter.writerow(to_write)


                