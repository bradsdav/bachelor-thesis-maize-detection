import csv

acc1 = []
loss1 = []

f = open('results/eval/tested_results.txt', 'r')
lines = f.readlines()
for i in range(len(lines)):
    if i % 2 == 1:
        acc1.append(float(lines[i]))
    else:
        loss1.append(float(lines[i]))
f.close()

acc2 = []
loss2 = []

f = open('results/eval/tested_results_hid.txt', 'r')
lines = f.readlines()
for i in range(len(lines)):
    if i % 2 == 1:
        acc2.append(float(lines[i]))
    else:
        loss2.append(float(lines[i]))
f.close()

acc3 = []
loss3 = []

f = open('results/eval/data_spec.txt', 'r')
lines = f.readlines()
for i in range(len(lines)):
    if i % 2 == 1:
        acc3.append(float(lines[i]))
    else:
        loss3.append(float(lines[i]))
f.close()

x = []
for i in range(12):
    x.append(100*((i*7)/73))
x[11] = 100

with open('results/eval/accuracy_sparsity.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    to_write = []
    for i in range(len(x)):
        to_write.append(x[i])
        to_write.append(acc1[i])
        
    writer.writerow(to_write)

    to_write = []
    for i in range(len(x)):
        to_write.append(x[i])
        to_write.append(acc2[i])
        
    writer.writerow(to_write)

    to_write = []
    for i in range(len(x)):
        to_write.append(x[i])
        to_write.append(acc3[i])
        
    writer.writerow(to_write)


