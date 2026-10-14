import os
import numpy as np
from src import dataset
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Bottleneck, ResNet, ResNet50
from src.model_dropout import Bottleneck_drop, ResNet_drop, ResNet50_drop

print("ok")

batch_size = 200
epochs = 250

trainingDataset = dataset.myDataset('./data/confidence/train_pred_1.pkl', './data/confidence/train_labels_pred_1.pkl')
validationDataset = dataset.myDataset('./data/orig/validation_or.pkl', './data/orig/validation_labels_or.pkl')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainloader = torch.utils.data.DataLoader(trainingDataset, batch_size=batch_size, shuffle=True, num_workers=6)
valloader = torch.utils.data.DataLoader(validationDataset, batch_size=batch_size, shuffle=True, num_workers=6)



f = open('results/logs/run_pred.txt', 'w')
f.close()

f = open('results/logs/losses_pred.txt', 'w')
f.close()

f = open('results/logs/accuracy_pred.txt', 'w')
f.close()

    
net = ResNet50(2).to(device)
best_run = 0

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)

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

        hid = random.randint(0, 73)
        mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        input = input * mask

        output = net(input)

        # print(output)
        # print(label)
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        cur_loss_train += loss.item()

        for k in range(len(output)):
            if output[k][0] > output[k][1]:
                if label[k][0] == 1:
                    train_accuracy += 1
            else:
                if label[k][1] == 1:
                    train_accuracy += 1
        
        tr_len += len(output)

        # if i % 100 == 0 and i != 0:
        #     print("Loss", i//1000, ":", cur_loss_train/tr_len)
        #     print('accuracy', 100*train_accuracy/tr_len)
        #     total_loss_train += cur_loss_train
        #     cur_loss_train = 0

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

        hid = random.randint(0, 73)
        mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        input = input * mask


        with torch.no_grad():
            output = net(input)
            loss = criterion(output, label)
            total_loss_val += loss.item()
        
        for k in range(len(output)):
            if output[k][0] > output[k][1]:
                if label[k][0] == 1:
                    val_accuracy += 1
            else:
                if label[k][1] == 1:
                    val_accuracy += 1
        
        val_len += len(output)
            
    
    print("Total validation loss hid:", total_loss_val/val_len)
    print('Validation accuracy hid:', 100*val_accuracy/val_len, '%')

    file = 'results/logs/run_pred.txt'
    f = open(file, 'a')
    f.write('Epoch ')
    f.write(str(epoch))
    f.write(':\nAccuracy:')
    f.write(str(100*val_accuracy/val_len))
    f.write("\nTotal validation loss:")
    f.write(str(total_loss_val/val_len))
    f.write("\n")
    f.close()

    file = 'results/logs/losses_pred.txt'
    f = open(file, 'a')
    f.write(str(total_loss_train/tr_len))
    f.write("\n")
    f.write(str(total_loss_val/val_len))
    f.write("\n")
    f.close()

    file = 'results/logs/accuracy_pred.txt'
    f = open(file, 'a')
    f.write(str(100*train_accuracy/tr_len))
    f.write("\n")
    f.write(str(100*val_accuracy/val_len))
    f.write("\n")
    f.close()

    
    if epoch == 0 or val_accuracy > best_run:
        best_run = val_accuracy
        file = 'weights/weights_pred_1.pts'
        torch.save(net.state_dict(), file)
        print("Updating weights")

        file = 'results/logs/run_pred.txt'
        f = open(file, 'a')
        f.write('Updating weights\n')
        f.close()

    file = 'results/logs/run_pred.txt'
    f = open(file, 'a')
    f.write("\n")
    f.close()

print('Training hidden Done')























print("ok")

batch_size = 200
epochs = 150

trainingDataset = dataset.myDataset('./data/confidence/train_pred_1.pkl', './data/confidence/train_labels_pred_1.pkl')
validationDataset = dataset.myDataset('./data/orig/validation_or.pkl', './data/orig/validation_labels_or.pkl')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainloader = torch.utils.data.DataLoader(trainingDataset, batch_size=batch_size, shuffle=True, num_workers=6)
valloader = torch.utils.data.DataLoader(validationDataset, batch_size=batch_size, shuffle=True, num_workers=6)




f = open('results/logs/run_pred_dr.txt', 'w')
f.close()

f = open('results/logs/losses_pred_dr.txt', 'w')
f.close()

f = open('results/logs/accuracy_pred_dr.txt', 'w')
f.close()

    
net = ResNet50_drop(2)
net.load_state_dict(torch.load('./weights/weights_pred_1.pts', map_location='cpu'))
net = net.to(device)

best_run = 0

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)

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

        hid = random.randint(0, 73)
        mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        input = input * mask

        output = net(input)

        # print(output)
        # print(label)
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        cur_loss_train += loss.item()

        for k in range(len(output)):
            if output[k][0] > output[k][1]:
                if label[k][0] == 1:
                    train_accuracy += 1
            else:
                if label[k][1] == 1:
                    train_accuracy += 1
        
        tr_len += len(output)

        # if i % 100 == 0 and i != 0:
        #     print("Loss", i//1000, ":", cur_loss_train/tr_len)
        #     print('accuracy', 100*train_accuracy/tr_len)
        #     total_loss_train += cur_loss_train
        #     cur_loss_train = 0

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

        hid = random.randint(0, 73)
        mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        input = input * mask


        with torch.no_grad():
            output = net(input)
            loss = criterion(output, label)
            total_loss_val += loss.item()
        
        for k in range(len(output)):
            if output[k][0] > output[k][1]:
                if label[k][0] == 1:
                    val_accuracy += 1
            else:
                if label[k][1] == 1:
                    val_accuracy += 1
        
        val_len += len(output)
            
    
    print("Total validation loss:", total_loss_val/val_len)
    print('Validation accuracy:', 100*val_accuracy/val_len, '%')

    file = 'results/logs/run_pred_dr.txt'
    f = open(file, 'a')
    f.write('Epoch ')
    f.write(str(epoch))
    f.write(':\nAccuracy:')
    f.write(str(100*val_accuracy/val_len))
    f.write("\nTotal validation loss:")
    f.write(str(total_loss_val/val_len))
    f.write("\n")
    f.close()

    file = 'results/logs/losses_pred_dr.txt'
    f = open(file, 'a')
    f.write(str(total_loss_train/tr_len))
    f.write("\n")
    f.write(str(total_loss_val/val_len))
    f.write("\n")
    f.close()

    file = 'results/logs/accuracy_pred_dr.txt'
    f = open(file, 'a')
    f.write(str(100*train_accuracy/tr_len))
    f.write("\n")
    f.write(str(100*val_accuracy/val_len))
    f.write("\n")
    f.close()

    
    if epoch == 0 or val_accuracy > best_run:
        best_run = val_accuracy
        file = 'weights/weights_pred_dr_1.pts'
        torch.save(net.state_dict(), file)
        print("Updating weights")

        file = 'results/logs/run_pred_dr.txt'
        f = open(file, 'a')
        f.write('Updating weights\n')
        f.close()

    file = 'results/logs/run_pred_dr.txt'
    f = open(file, 'a')
    f.write("\n")
    f.close()

print('Training hidden Done')





















print("ok")

batch_size = 200
epochs = 350

trainingDataset = dataset.myDataset('./data/confidence/train_conf_1.pkl', './data/confidence/train_labels_conf_1.pkl')
validationDataset = dataset.myDataset('./data/orig/validation_or.pkl', './data/orig/validation_labels_or.pkl')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainloader = torch.utils.data.DataLoader(trainingDataset, batch_size=batch_size, shuffle=True, num_workers=6)
valloader = torch.utils.data.DataLoader(validationDataset, batch_size=batch_size, shuffle=True, num_workers=6)

    
pred = ResNet50(2)
pred.load_state_dict(torch.load('./weights/weights_pred_dr_1.pts', map_location='cpu'))
pred = pred.to(device)

conf = ResNet50_drop(2).to(device)

best_run = 0

learning_rate = 0.00001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(conf.parameters(), lr=learning_rate)

file = 'results/logs/run_conf.txt'
f = open(file, 'w')
f.close()

file = 'results/logs/accuracy_conf.txt'
f = open(file, 'w')
f.close()


for epoch in range(epochs):
    total_loss_train = 0
    cur_loss_train = 0

    total_loss_val = 0
    
    pred_acc = 0
    conf_acc = 0

    val_pred_acc = 0
    val_conf_acc = 0

    tr_len = 0
    val_len = 0

    for i, data in enumerate(trainloader):
        input = data['data'].to(device)
        label = data['label'].to(device)


        hid = random.randint(0, 73)
        mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        input = input * mask

        with torch.no_grad():
            pred_out = pred(input)

        conf_out = conf(input)

        targ = torch.zeros(input.size(0), 2).to(device)


        for k in range(len(pred_out)):
            if torch.argmax(pred_out[k]) == torch.argmax(label[k]):
                targ[k][0] = 1
            else:
                targ[k][1] = 1

        
        loss = criterion(conf_out, targ)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        cur_loss_train += loss.item()

        for k in range(len(pred_out)):
            if torch.argmax(pred_out[k]) == torch.argmax(label[k]):
                pred_acc += 1

        for k in range(len(conf_out)):
            if torch.argmax(conf_out[k]) == torch.argmax(targ[k]):
                conf_acc += 1

        tr_len += len(conf_out)

        # if i % 100 == 0:
        #     print("Loss", i//100, ":", cur_loss_train)
        #     print('Pred_acc:', 100*pred_acc/tr_len)
        #     print('Conf_acc:', 100*conf_acc/tr_len)
        #     total_loss_train += cur_loss_train
        #     cur_loss_train = 0



    for i, data in enumerate(valloader):
        if i == 250:
            break

        input = data['data'].to(device)
        label = data['label'].to(device)
        # label = torch.reshape(label, (-1,))

        hid = random.randint(0, 73)
        mask = torch.cat((torch.ones(input.size(0), 10, 73-hid), torch.zeros(input.size(0), 10, hid)), 2).to(device)
        input = input * mask

        with torch.no_grad():
            pred_out = pred(input)
            conf_out = conf(input)

        targ = torch.zeros(input.size(0), 2).to(device)


        for k in range(len(pred_out)):
            if torch.argmax(pred_out[k]) == torch.argmax(label[k]):
                targ[k][0] = 1
            else:
                targ[k][1] = 1

        
        loss = criterion(conf_out, targ)
        total_loss_val += loss.item()



        for k in range(len(pred_out)):
            if torch.argmax(pred_out[k]) == torch.argmax(label[k]):
                val_pred_acc += 1

        for k in range(len(conf_out)):
            if torch.argmax(conf_out[k]) == torch.argmax(targ[k]):
                val_conf_acc += 1

        val_len += len(conf_out)




    print('Epoch:', epoch+1)
    total_loss_train += cur_loss_train
    print("Total training loss:", total_loss_train)
    print("Prediction accuracy:", 100*pred_acc/tr_len)
    print("Confidence accuracy:", 100*conf_acc/tr_len)

    print("Valid loss:", total_loss_val)
    print("Valid prediction accuracy:", 100*val_pred_acc/val_len)
    print("Valid confidence accuracy:", 100*val_conf_acc/val_len)



    file = 'results/logs/run_conf.txt'
    f = open(file, 'a')
    f.write('Epoch ')
    f.write(str(epoch))
    f.write(':\nAccuracy:')
    f.write(str(100*val_conf_acc/val_len))
    f.write("\n")
    f.close()


    file = 'results/logs/accuracy_conf.txt'
    f = open(file, 'a')
    f.write(str(100*conf_acc/tr_len))
    f.write("\n")
    f.write(str(100*val_conf_acc/val_len))
    f.write("\n")
    f.close()

    
    if epoch == 0 or val_conf_acc > best_run:
        best_run = val_conf_acc
        file = 'weights/weights_conf_1.pts'
        torch.save(conf.state_dict(), file)
        print("Updating weights")

        file = 'results/logs/run_conf.txt'
        f = open(file, 'a')
        f.write('Updating weights\n')
        f.close()

    file = 'results/logs/run_conf.txt'
    f = open(file, 'a')
    f.write("\n")
    f.close()

print('Training confidence done')



