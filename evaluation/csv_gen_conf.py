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


testDataset = dataset.myDataset('./data/orig/test_or.pkl', './data/orig/test_labels_or.pkl')

testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=6)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pred = ResNet50(2)
pred.load_state_dict(torch.load('./weights/weights_pred_dr.pts', map_location='cpu'))
pred = pred.to(device)

conf = ResNet50(2)
conf.load_state_dict(torch.load('./weights/weights_conf_1.pts', map_location='cpu'))
conf = conf.to(device)




print("generating test")                

with open('results/eval/test_conf.csv', 'w', newline='') as csvfile:
    testwriter = csv.writer(csvfile)
    for i, data in enumerate(testloader):
        input = data['data'].to(device)
        label = data['label'].to(device)


        with torch.no_grad():
            pred_out = pred(input)
            conf_out = conf(input)
            to_write = [0,0,0]
            for k in range(len(pred_out)):
                to_write[0] = pred_out[k][0].item()
                if label[k][0] == 1:
                    to_write[1] = 1
                else:
                    to_write[1] = 0
                to_write[2] = conf_out[k][0].item()

                testwriter.writerow(to_write)

print("generating test hid")                

with open('results/eval/test_conf_rand_hid.csv', 'w', newline='') as csvfile:
    testwriter = csv.writer(csvfile)
    for i, data in enumerate(testloader):
        input = data['data'].to(device)
        label = data['label'].to(device)

        hid = random.randint(0, 73)
        mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        input = input * mask
        perc = 100*(hid/73)

        with torch.no_grad():
            pred_out = pred(input)
            conf_out = conf(input)
            to_write = [0,0,0,perc]
            for k in range(len(pred_out)):
                to_write[0] = pred_out[k][0].item()
                if label[k][0] == 1:
                    to_write[1] = 1
                else:
                    to_write[1] = 0
                to_write[2] = conf_out[k][0].item()

                testwriter.writerow(to_write)


                