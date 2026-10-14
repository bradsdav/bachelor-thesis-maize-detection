import os
import numpy as np
from src import dataset_miss
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Bottleneck, ResNet, ResNet50

print("ok")

batch_size = 200

testDataset = dataset_miss.myDataset('./data/orig/test_miss.pkl', './data/orig/test_labels_miss.pkl')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ResNet50(2, 9)
net.load_state_dict(torch.load('./weights/weights_miss_dr.pts', map_location='cpu'))
net = net.to(device)

criterion = nn.BCELoss()

testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=6)

print("Testing")

total_loss_test = 0


mask = torch.ones(batch_size, 9, 73)
mask = mask.to(device)

f = open('results/eval/tested_results_miss.txt', 'w')
f.close()

for j in range(12):
    total_loss_test = 0
    accuracy = 0
    test_len = 0
    
    for i, data in enumerate(testloader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        # label = torch.reshape(label, (-1,))
        
        if j == 11:
            mask = torch.zeros(input.size(0), 9, 73).to(device)
        else:
            hid = j*7
            mask = torch.cat((torch.ones(input.size(0), 9, 73-hid), torch.zeros(input.size(0), 9, hid)), 2).to(device)
        input = input * mask
        test_len += input.size(0)

        with torch.no_grad():
            output = net(input)
            # print(output)
            # print(label)
            loss = criterion(output, label)
            total_loss_test += loss.item()

        for k in range(len(output)):
            if output[k][0] > output[k][1]:
                if label[k][0] == 1:
                    accuracy += 1
            else:
                if label[k][1] == 1:
                    accuracy += 1
        
        

    print('Average loss with', j*7, 'positions hidden:', total_loss_test/test_len)
    print('Accuracy with', j*7, 'positions hidden:', 100*accuracy/test_len)
    print('')

    f = open('results/eval/tested_results_miss.txt', 'a')
    f.write(str(total_loss_test/test_len))
    f.write("\n")
    f.write(str(100*accuracy/test_len))
    f.write("\n")
    f.close()

            
