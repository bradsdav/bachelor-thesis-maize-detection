import os
import numpy as np
from src import dataset
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Bottleneck, ResNet, ResNet50

print("ok")

batch_size = 200

testDataset = dataset.myDataset('./data/rand/test.pkl', './data/rand/test_labels.pkl')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ResNet50(2)
net.load_state_dict(torch.load('./weights/weights_hid_dr.pts', map_location='cpu'))
net = net.to(device)

criterion = nn.BCELoss()

testloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=6)

print("Confidence")


# mask = torch.ones(batch_size, 10, 73)
# mask = mask.to(device)

for x in range(11):

    num = dict()
    acc = dict()
    for i in range(50, 101):
        num[str(i)] = 0
        acc[str(i)] = 0

    for i, data in enumerate(testloader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        # label = torch.reshape(label, (-1,))

        hid = x*7
        mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)

        input = input * mask

        with torch.no_grad():
            output = net(input)

            labels = []
            for j in range(len(label)):
                if label[j][0] == 1:
                    labels.append(0)
                else:
                    labels.append(1)

            for j in range(len(output)):
                if output[j][0] > output[j][1]:
                    if output[j][0] == 1:
                        num['100'] += 1
                        if labels[j] == 0:
                            acc['100'] += 1
                        continue
                    k = 1
                    while k > 0:
                        k -= 0.01
                        if output[j][0].item() > k:
                            # if int(100*k) == 49:
                            #     print(output[j])
                            #     print(labels[j])
                            #     print(0)

                            num[str(int(100*(k+0.0000001)))] += 1
                            if labels[j] == 0:
                                acc[str(int(100*(k+0.0000001)))] += 1
                            break

                else:
                    if output[j][1] == 1:
                        num['100'] += 1
                        if labels[j] == 1:
                            acc['100'] += 1
                        continue
                    k = 1
                    while k > 0:
                        k -= 0.01
                        if output[j][1].item() > k:
                            # if int(100*k) == 49:
                            #     print(output[j])
                            #     print(labels[j])
                            #     print(1)

                            num[str(int(100*(k+0.0000001)))] += 1
                            if labels[j] == 1:
                                acc[str(int(100*(k+0.0000001)))] += 1
                            break


    print('fin', x*7)
    for i in range(100, 50, -1):
        num[str(i-1)] += num[str(i)]
        acc[str(i-1)] += acc[str(i)]

    f = open('results/eval/conf_num_' + str(x) + '.txt', 'w')
    for i in range(100, 49, -1):
        f.write(str(num[str(i)]))
        f.write("\n")
    f.close()

    f = open('results/eval/conf_acc_' + str(x) + '.txt', 'w')
    for i in range(100, 49, -1):
        f.write(str(acc[str(i)]))
        f.write("\n")
    f.close()

