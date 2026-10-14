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
from src.model_dropout import Bottleneck_drop, ResNet_drop, ResNet50_drop

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

net_drop = ResNet50_drop(2)
net_drop.load_state_dict(torch.load('./weights/weights_hid_or_dr.pts', map_location='cpu'))
net_drop = net_drop.to(device)

criterion = nn.BCELoss()


# print("generating val")

# to_write = []
# for x in range(batch_size):
#     to_write.append([])
#     for z in range(12):
#         to_write[x].append(0)

# with open('results/eval/val_drop.csv', 'w', newline='') as csvfile:
#     valwriter = csv.writer(csvfile)
#     for i, data in enumerate(valloader):
#         input = data['data'].to(device)
#         label = data['label'].to(device)
#         # label = torch.reshape(label, (-1,))

#         if input.size(0) != batch_size:
#             to_write = []
#             for x in range(input.size(0)):
#                 to_write.append([])
#                 for z in range(12):
#                     to_write[x].append(0)
            

#         with torch.no_grad():
            
#             # hid = random.randint(0, 73)
#             # mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
#             # input = input * mask
#             # perc = 100*(hid/73)

#             for l in range(10):
#                 output = net_drop(input)
#                 for k in range(len(output)):
#                     to_write[k][l] = output[k][0].item()
            
#             output = net(input)
#             for k in range(len(output)):
#                 to_write[k][11] = output[k][0].item()

#             for k in range(len(output)):
#                 if label[k][0] == 1:
#                     to_write[k][10] = 1
#                 else:
#                     to_write[k][10] = 0
#                 # to_write[k][11] = perc
#                 valwriter.writerow(to_write[k])


print("generating val")

to_write = []
for x in range(batch_size):
    to_write.append([])
    for z in range(12):
        to_write[x].append(0)

with open('results/eval/val_drop.csv', 'w', newline='') as csvfile:
    testwriter = csv.writer(csvfile)
    for i, data in enumerate(valloader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        # label = torch.reshape(label, (-1,))

        if input.size(0) != batch_size:
            to_write = []
            for x in range(input.size(0)):
                to_write.append([])
                for z in range(12):
                    to_write[x].append(0)


        with torch.no_grad():
            
            # hid = random.randint(0, 73)
            # mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
            # input = input * mask
            # perc = 100*(hid/73)

            for l in range(10):
                output = net_drop(input)
                for k in range(len(output)):
                    to_write[k][l] = output[k][0].item()
            
            output = net(input)
            for k in range(len(output)):
                to_write[k][11] = output[k][0].item()

            for k in range(len(output)):
                if label[k][0] == 1:
                    to_write[k][10] = 1
                else:
                    to_write[k][10] = 0
                # to_write[k][11] = perc
                testwriter.writerow(to_write[k])


print("generating val hid")

to_write = []
for x in range(batch_size):
    to_write.append([])
    for z in range(13):
        to_write[x].append(0)

with open('results/eval/val_drop_rand_hid.csv', 'w', newline='') as csvfile:
    testwriter = csv.writer(csvfile)
    for i, data in enumerate(valloader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        # label = torch.reshape(label, (-1,))

        if input.size(0) != batch_size:
            to_write = []
            for x in range(input.size(0)):
                to_write.append([])
                for z in range(13):
                    to_write[x].append(0)


        with torch.no_grad():
            
            hid = random.randint(0, 73)
            mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
            input = input * mask
            perc = 100*(hid/73)

            for l in range(10):
                output = net_drop(input)
                for k in range(len(output)):
                    to_write[k][l] = output[k][0].item()
            
            output = net(input)
            for k in range(len(output)):
                to_write[k][12] = output[k][0].item()

            for k in range(len(output)):
                if label[k][0] == 1:
                    to_write[k][10] = 1
                else:
                    to_write[k][10] = 0
                to_write[k][11] = perc
                testwriter.writerow(to_write[k])
