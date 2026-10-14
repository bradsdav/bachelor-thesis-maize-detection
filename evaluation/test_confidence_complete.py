import csv
import random
import os
import numpy as np
from src import dataset
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.model import Bottleneck, ResNet, ResNet50
from src.model_dropout import Bottleneck_drop, ResNet_drop, ResNet50_drop

batch_size = 100

testDataset = dataset.myDataset('./data/rand/test.pkl', './data/rand/test_labels.pkl')

testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=6)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ResNet50(2)
net.load_state_dict(torch.load('./weights/weights_robust.pts', map_location='cpu'))
net = net.to(device)

rows = []

for i, data in enumerate(testloader):
    input = data['data'].to(device)
    label = data['label'].to(device)

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
            rows.append(to_write)

softmax = []
for i in range(len(rows)):
    if rows[i][0] > rows[i][1] and rows[i][2] == 0 or rows[i][0] < rows[i][1] and rows[i][2] == 1:
        log = 1
    else:
        log = 0
    softmax.append([2*max(rows[i][0], rows[i][1]) - 1, log])

def key(x):
    return x[0]

softmax.sort(reverse= True, key = key)

for i in range(1, len(softmax)):
    softmax[i][1] += softmax[i-1][1]

for i in range(len(softmax)):
    softmax[i][1] = 100*(softmax[i][1]/(i+1))









net_drop = ResNet50_drop(2)
net_drop.load_state_dict(torch.load('./weights/weights_robust.pts', map_location='cpu'))
net_drop = net_drop.to(device)

to_write = []
for x in range(batch_size):
    to_write.append([])
    for z in range(12):
        to_write[x].append(0)

rows = []

for i, data in enumerate(testloader):
        input = data['data'].to(device)
        label = data['label'].to(device)

        if input.size(0) != batch_size:
            to_write = []
            for x in range(input.size(0)):
                to_write.append([])
                for z in range(12):
                    to_write[x].append(0)


        with torch.no_grad():

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
                rows.append(to_write[k])

drop_data = []
data = []
for row in rows:
    x = []
    for i in range(10):
        x.append(float(row[i]))
    drop_data.append(x)

    x = []
    for i in range(10, 12):
        x.append(float(row[i]))
    data.append(x)

drop = []
for i in range(len(data)):
    if abs(data[i][0] - data[i][1]) < 0.5:
        log = 1
    else:
        log = 0
    drop.append([1 - 10*np.var(drop_data[i]), log])

def key(x):
    return x[0]

drop.sort(reverse= True, key = key)

for i in range(1, len(softmax)):
    drop[i][1] += drop[i-1][1]

for i in range(len(softmax)):
    drop[i][1] = 100*(drop[i][1]/(i+1))









pred = ResNet50(2)
pred.load_state_dict(torch.load('./weights/weights_robust.pts', map_location='cpu'))
pred = pred.to(device)

conf = ResNet50(2)
conf.load_state_dict(torch.load('./weights/weights_confidence.pts', map_location='cpu'))
conf = conf.to(device)

rows = []

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

            rows.append(to_write)

confidence = []
for i in range(len(rows)):
    if rows[i][0] > 0.5 and rows[i][1] == 1 or rows[i][0] < 0.5 and rows[i][1] == 0:
        log = 1
    else:
        log = 0
    confidence.append([rows[i][2], log])

def key(x):
    return x[0]

confidence.sort(reverse= True, key = key)

for i in range(1, len(confidence)):
    confidence[i][1] += confidence[i-1][1]

for i in range(len(confidence)):
    confidence[i][1] = 100*(confidence[i][1]/(i+1))










y1 = []
y2 = []
y3 = []
x1 = []
x2 = []
x3 = []
for i in range(len(confidence)):
    y1.append(softmax[i][1])
    y2.append(drop[i][1])
    y3.append(confidence[i][1])

    x1.append(softmax[i][0])
    x2.append(drop[i][0])
    x3.append(confidence[i][0])

x = []
for i in range(len(softmax)):
    x.append(100*(i+1)/len(softmax))

plt.plot(x, y1, 'g-', label = 'C_s')
plt.plot(x, y2, 'r-', label = 'C_d')
plt.plot(x, y3, 'c-', label = 'C_cn')

# naming the x axis
plt.xlabel('coverage [%]')
# naming the y axis
plt.ylabel('accuracy [%]')

plt.grid()

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()



plt.plot(x1, y1, 'g-', label = 'C_s')
plt.plot(x2, y2, 'r-', label = 'C_d')
plt.plot(x3, y3, 'c-', label = 'C_cn')

# grab a reference to the current axes
ax = plt.gca()
# set the xlimits to be the reverse of the current xlimits
ax.set_xlim(ax.get_xlim()[::-1])
# call `draw` to re-render the graph
plt.draw()

# naming the x axis
plt.xlabel('confidence')
# naming the y axis
plt.ylabel('accuracy [%]')

plt.grid()

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()

