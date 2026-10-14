import os
import numpy as np
from src import dataset_16
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Bottleneck, ResNet, ResNet50
from src.model_dropout import Bottleneck_drop, ResNet_drop, ResNet50_drop

print("ok")

batch_size = 200
epochs = 150

trainingDataset = dataset_16.myDataset('./data/orig/train_16.pkl', './data/orig/train_labels_16.pkl')
validationDataset = dataset_16.myDataset('./data/orig/validation_16.pkl', './data/orig/validation_labels_16.pkl')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainloader = torch.utils.data.DataLoader(trainingDataset, batch_size=batch_size, shuffle=True, num_workers=6)
valloader = torch.utils.data.DataLoader(validationDataset, batch_size=batch_size, shuffle=True, num_workers=6)


    
net = ResNet50(16).to(device)
best_run = 0

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(epochs):
    print('Epoch hidden:', epoch+1)
    total_loss_train = 0
    cur_loss_train = 0
    train_accuracy = 0

    tr_len = 0

    for i, data in enumerate(trainloader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        # label = torch.reshape(label, (-1,))

        # hid = random.randint(0, 73)
        # mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        # input = input * mask

        output = net(input)

        # print(output)
        # print(label)
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        cur_loss_train += loss.item()

        for k in range(len(output)):
            if torch.argmax(output[k]) == torch.argmax(label[k]):
                train_accuracy += 1

            # if output[k][0] > output[k][1]:
            #     if label[k][0] == 1:
            #         train_accuracy += 1
            # else:
            #     if label[k][1] == 1:
            #         train_accuracy += 1
        
        tr_len += len(output)

    total_loss_train += cur_loss_train
    print("Total training loss hid:", total_loss_train/tr_len)
    print('Training accuracy hid: ', 100*train_accuracy/tr_len, '%')


    total_loss_val = 0
    val_accuracy = 0
    val_len = 0

    for i, data in enumerate(valloader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        # label = torch.reshape(label, (-1,))

        # hid = random.randint(0, 73)
        # mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        # input = input * mask


        with torch.no_grad():
            output = net(input)
            loss = criterion(output, label)
            total_loss_val += loss.item()
        
        for k in range(len(output)):
            if torch.argmax(output[k]) == torch.argmax(label[k]):
                val_accuracy += 1

            # if output[k][0] > output[k][1]:
            #     if label[k][0] == 1:
            #         val_accuracy += 1
            # else:
            #     if label[k][1] == 1:
            #         val_accuracy += 1
        
        val_len += len(output)
            
    
    print("Total validation loss hid:", total_loss_val/val_len)
    print('Validation accuracy hid:', 100*val_accuracy/val_len, '%')

    file = 'results/logs/run_16.txt'
    f = open(file, 'a')
    f.write('Epoch ')
    f.write(str(epoch))
    f.write(':\nAccuracy:')
    f.write(str(100*val_accuracy/val_len))
    f.write("\nTotal validation loss:")
    f.write(str(total_loss_val/val_len))
    f.write("\n")
    f.close()

    file = 'results/logs/accuracy_16.txt'
    f = open(file, 'a')
    f.write(str(100*train_accuracy/tr_len))
    f.write("\n")
    f.write(str(100*val_accuracy/val_len))
    f.write("\n")
    f.close()

    
    if epoch == 0 or val_accuracy > best_run:
        best_run = val_accuracy
        file = 'weights/weights_16.pts'
        torch.save(net.state_dict(), file)
        print("Updating weights")

        file = 'results/logs/run_16.txt'
        f = open(file, 'a')
        f.write('Updating weights\n')
        f.close()

    file = 'results/logs/run_16.txt'
    f = open(file, 'a')
    f.write("\n")
    f.close()

print('Training hidden Done')























print("ok")

batch_size = 200
epochs = 100

trainingDataset = dataset_16.myDataset('./data/orig/train_16.pkl', './data/orig/train_labels_16.pkl')
validationDataset = dataset_16.myDataset('./data/orig/validation_16.pkl', './data/orig/validation_labels_16.pkl')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainloader = torch.utils.data.DataLoader(trainingDataset, batch_size=batch_size, shuffle=True, num_workers=6)
valloader = torch.utils.data.DataLoader(validationDataset, batch_size=batch_size, shuffle=True, num_workers=6)

    

net = ResNet50_drop(16)
net.load_state_dict(torch.load('./weights/weights_16.pts', map_location='cpu'))
net = net.to(device)

best_run = 0

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(epochs):
    print('Epoch hidden:', epoch+1)
    total_loss_train = 0
    cur_loss_train = 0
    train_accuracy = 0

    tr_len = 0

    for i, data in enumerate(trainloader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        # label = torch.reshape(label, (-1,))

        # hid = random.randint(0, 73)
        # mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        # input = input * mask

        output = net(input)

        # print(output)
        # print(label)
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        cur_loss_train += loss.item()

        for k in range(len(output)):
            if torch.argmax(output[k]) == torch.argmax(label[k]):
                train_accuracy += 1

            # if output[k][0] > output[k][1]:
            #     if label[k][0] == 1:
            #         train_accuracy += 1
            # else:
            #     if label[k][1] == 1:
            #         train_accuracy += 1
        
        tr_len += len(output)

    total_loss_train += cur_loss_train
    print("Total training loss:", total_loss_train/tr_len)
    print('Training accuracy: ', 100*train_accuracy/tr_len, '%')


    total_loss_val = 0
    val_accuracy = 0
    val_len = 0

    for i, data in enumerate(valloader):
        input = data['data'].to(device)
        label = data['label'].to(device)
        # label = torch.reshape(label, (-1,))

        # hid = random.randint(0, 73)
        # mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        # input = input * mask


        with torch.no_grad():
            output = net(input)
            loss = criterion(output, label)
            total_loss_val += loss.item()
        
        for k in range(len(output)):
            if torch.argmax(output[k]) == torch.argmax(label[k]):
                val_accuracy += 1

            # if output[k][0] > output[k][1]:
            #     if label[k][0] == 1:
            #         val_accuracy += 1
            # else:
            #     if label[k][1] == 1:
            #         val_accuracy += 1
        
        val_len += len(output)
            
    
    print("Total validation loss:", total_loss_val/val_len)
    print('Validation accuracy:', 100*val_accuracy/val_len, '%')

    file = 'results/logs/run_16_dr.txt'
    f = open(file, 'a')
    f.write('Epoch ')
    f.write(str(epoch))
    f.write(':\nAccuracy:')
    f.write(str(100*val_accuracy/val_len))
    f.write("\nTotal validation loss:")
    f.write(str(total_loss_val/val_len))
    f.write("\n")
    f.close()

    file = 'results/logs/accuracy_16_dr.txt'
    f = open(file, 'a')
    f.write(str(100*train_accuracy/tr_len))
    f.write("\n")
    f.write(str(100*val_accuracy/val_len))
    f.write("\n")
    f.close()

    
    if epoch == 0 or val_accuracy > best_run:
        best_run = val_accuracy
        file = 'weights/weights_16_dr.pts'
        torch.save(net.state_dict(), file)
        print("Updating weights")

        file = 'results/logs/run_16_dr.txt'
        f = open(file, 'a')
        f.write('Updating weights\n')
        f.close()

    file = 'results/logs/run_16_dr.txt'
    f = open(file, 'a')
    f.write("\n")
    f.close()

print('Training hidden Done')

