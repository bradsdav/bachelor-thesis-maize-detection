import os
import numpy as np
from src import dataset
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.model import Bottleneck, ResNet, ResNet50

batch_size = 200

testDataset = dataset.myDataset('./data/rand/test.pkl', './data/rand/test_labels.pkl')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ResNet50(2)
net.load_state_dict(torch.load('./weights/weights_baseline.pts', map_location='cpu'))
net = net.to(device)

criterion = nn.BCELoss()

testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=6)

print("Testing baseline network!")

mask = torch.ones(batch_size, 10, 73)
mask = mask.to(device)

acc1 = []

for j in range(12):
    accuracy = 0
    test_len = 0
    
    for i, data in enumerate(testloader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        
        if j == 11:
            mask = torch.zeros(input.size(0), 10, 73).to(device)
        else:
            hid = j*7
            mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        input = input * mask
        test_len += input.size(0)

        with torch.no_grad():
            output = net(input)

        for k in range(len(output)):
            if output[k][0] > output[k][1]:
                if label[k][0] == 1:
                    accuracy += 1
            else:
                if label[k][1] == 1:
                    accuracy += 1
        
    if j == 11:
        print('Accuracy with an empty input:', 100*accuracy/test_len)
    else:
        print('Accuracy with', j*7, 'positions hidden:', 100*accuracy/test_len)
    print('')

    acc1.append(100*accuracy/test_len)









net = ResNet50(2)
net.load_state_dict(torch.load('./weights/weights_robust.pts', map_location='cpu'))
net = net.to(device)

criterion = nn.BCELoss()

testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=6)

print("Testing robust network!")

mask = torch.ones(batch_size, 10, 73)
mask = mask.to(device)

acc2 = []

for j in range(12):
    accuracy = 0
    test_len = 0
    
    for i, data in enumerate(testloader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        
        if j == 11:
            mask = torch.zeros(input.size(0), 10, 73).to(device)
        else:
            hid = j*7
            mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        input = input * mask
        test_len += input.size(0)

        with torch.no_grad():
            output = net(input)

        for k in range(len(output)):
            if output[k][0] > output[k][1]:
                if label[k][0] == 1:
                    accuracy += 1
            else:
                if label[k][1] == 1:
                    accuracy += 1

    if j == 11:
        print('Accuracy with an empty input:', 100*accuracy/test_len)
    else:
        print('Accuracy with', j*7, 'positions hidden:', 100*accuracy/test_len)
    print('')

    acc2.append(100*accuracy/test_len)









testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=6)

print("Testing specialised networks!")

acc3 = []

for j in range(12):
    if j == 0 or j == 11:
        net = ResNet50(2)
        net.load_state_dict(torch.load('./weights/weights_baseline.pts', map_location='cpu'))
        net = net.to(device)
    else:
        net = ResNet50(2)
        net.load_state_dict(torch.load('./weights/weights_'+ str(j*7) +'.pts', map_location='cpu'))
        net = net.to(device)
        
    criterion = nn.BCELoss()

    total_loss_test = 0
    accuracy = 0
    test_len = 0
    
    for i, data in enumerate(testloader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        
        if j == 11:
            mask = torch.zeros(input.size(0), 10, 73).to(device)
        else:
            hid = j*7
            mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        input = input * mask
        test_len += input.size(0)

        with torch.no_grad():
            output = net(input)

        for k in range(len(output)):
            if output[k][0] > output[k][1]:
                if label[k][0] == 1:
                    accuracy += 1
            else:
                if label[k][1] == 1:
                    accuracy += 1
        
    if j == 11:
        print('Accuracy with an empty input:', 100*accuracy/test_len)
    else:
        print('Accuracy with', j*7, 'positions hidden:', 100*accuracy/test_len)
    print('')

    acc3.append(100*accuracy/test_len)









x = []
for i in range(12):
    x.append(100 - 100*(((11-i)*7)/73))
x[0] = 0

x.reverse()

plt.plot(x, acc1, 'r-', label = 'baseline model')
plt.plot(x, acc2, 'g-', label = 'robust model')
plt.plot(x, acc3, 'b-', label = 'specialised models')

# naming the x axis
plt.xlabel('completeness [%]')
# naming the y axis
plt.ylabel('accuracy [%]')

plt.grid()

# # giving a title to my graph
# plt.title('Accuracy on testing dataset with a certain amount of data hidden from the end')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()
