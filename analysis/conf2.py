import numpy as np
import csv
import matplotlib.pyplot as plt

test = open('results/eval/test_rand_hid.csv')

csvreader = csv.reader(test)

rows = []
for row in csvreader:
    x = [0,0,0,0]
    for i in range(len(row)):
        x[i] = float(row[i])
    rows.append(x)


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

print('ok')



completeness = []
for i in range(len(rows)):
    if rows[i][0] > rows[i][1] and rows[i][2] == 0 or rows[i][0] < rows[i][1] and rows[i][2] == 1:
        log = 1
    else:
        log = 0
    completeness.append([(100 - rows[i][3])/100, log])

def key(x):
    return x[0]

completeness.sort(reverse= True, key = key)

for i in range(1, len(completeness)):
    completeness[i][1] += completeness[i-1][1]

for i in range(len(completeness)):
    completeness[i][1] = 100*(completeness[i][1]/(i+1))

print('ok')



test_drop = open('results/eval/test_drop_rand_hid.csv')

csvreader = csv.reader(test_drop)

drop_data = []
data = []
for row in csvreader:
    x = []
    for i in range(10):
        x.append(float(row[i]))
    drop_data.append(x)

    x = []
    for i in range(10, 13):
        x.append(float(row[i]))
    data.append(x)

drop = []
for i in range(len(data)):
    if abs(data[i][0] - data[i][2]) < 0.5:
        log = 1
    else:
        log = 0
    drop.append([1 - 10*np.var(drop_data[i]), log])

def key(x):
    return x[0]

drop.sort(reverse= True, key = key)

for i in range(1, len(drop)):
    drop[i][1] += drop[i-1][1]

for i in range(len(drop)):
    drop[i][1] = 100*(drop[i][1]/(i+1))

print('ok')





conf = open('results/eval/test_conf_rand_hid.csv')

csvreader = csv.reader(conf)

rows = []
for row in csvreader:
    x = [0,0,0,0]
    for i in range(len(row)):
        x[i] = float(row[i])
    rows.append(x)


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

# if confidence[14][0] == confidence[300][0] and confidence[300][1] == 1:
#     confidence[14][1] = 1
#     confidence[300][1] = 0
# else:
#     print('ERROR')

for i in range(1, len(confidence)):
    confidence[i][1] += confidence[i-1][1]

for i in range(len(confidence)):
    confidence[i][1] = 100*(confidence[i][1]/(i+1))

print('ok')





y1 = []
y2 = []
y3 = []
y4 = []
x1 = []
x2 = []
x3 = []
x4 = []
for i in range(len(confidence)):
    y1.append(softmax[i][1])
    y2.append(drop[i][1])
    y3.append(completeness[i][1])
    y4.append(confidence[i][1])

    x1.append(softmax[i][0])
    x2.append(drop[i][0])
    x3.append(completeness[i][0])
    x4.append(confidence[i][0])

x = []
for i in range(len(softmax)):
    x.append(100*(i+1)/len(softmax))

plt.plot(x, y1, 'g-', label = 'C_s')
plt.plot(x, y2, 'r-', label = 'C_d')
plt.plot(x, y3, 'b-', label = 'C_c')
plt.plot(x, y4, 'c-', label = 'C_cn')

# # grab a reference to the current axes
# ax = plt.gca()
# # set the xlimits to be the reverse of the current xlimits
# ax.set_xlim(ax.get_xlim()[::-1])
# # call `draw` to re-render the graph
# plt.draw()

# naming the x axis
plt.xlabel('coverage [%]')
# naming the y axis
plt.ylabel('accuracy [%]')

plt.grid()

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()

