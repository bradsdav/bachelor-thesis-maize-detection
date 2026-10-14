import numpy as np
import csv
import matplotlib.pyplot as plt

bins = 100

test = open('results/eval/val.csv')

csvreader = csv.reader(test)

rows = []
for row in csvreader:
    x = [0,0,0]
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

x = []
for i in range(bins,0,-1):
    x.append(i/bins-(1/bins))

h1 = []

y1 = [0]
div = 0
j = 0
for i in range(0, len(x)):
    while j < len(softmax) and softmax[j][0] > x[i]:
        y1[i] += softmax[j][1]
        j += 1
        div += 1
    if div != 0:
        y1[i] /= div
        y1[i] *= 100
    y1.append(0)
    h1.append(div)
    div = 0
y1.pop()

print('ok')



test_drop = open('results/eval/val_drop.csv')

csvreader = csv.reader(test_drop)

drop_data = []
data = []
for row in csvreader:
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
    drop.append([1 - 9*np.var(drop_data[i]), log])

def key(x):
    return x[0]

drop.sort(reverse= True, key = key)

x = []
for i in range(bins,0,-1):
    x.append(i/bins-(1/bins))

h2 = []

y2 = [0]
div = 0
j = 0
for i in range(0, len(x)):
    while j < len(drop) and drop[j][0] > x[i]:
        y2[i] += drop[j][1]
        j += 1
        div += 1
    if div != 0:
        y2[i] /= div
        y2[i] *= 100
    y2.append(0)
    h2.append(div)
    div = 0
y2.pop()

print('ok')



fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

x = []
for i in range(bins,0,-1):
    x.append(i/bins-((1/bins)/2))
  
# plotting a bar chart
ax2.bar(x, h1, width = 0.002, color = 'green', align='edge', label='C_s')

x = []
for i in range(bins,0,-1):
    x.append(i/bins-((1/bins)/2) + 0.003)
  
# plotting a bar chart
ax2.bar(x, h2, width = 0.002, color = 'red', align='edge', label='C_d')




x = []
for i in range(bins,0,-1):
    x.append(i/bins-((1/bins)/2))

i = 0
while i < len(x):
    if h1[i] == 0:
        x.pop(i)
        y1.pop(i)
    else:
        i += 1

# plotting points as a scatter plot
ax1.scatter(x, y1, color= "green", marker= "x", s=30, label='C_s')


x = []
for i in range(bins,0,-1):
    x.append(i/bins-((1/bins)/2))

i = 0
while i < len(x):
    if h2[i] == 0:
        x.pop(i)
        y2.pop(i)
        h2.pop(i)
    else:
        i += 1

# plotting points as a scatter plot
ax1.scatter(x, y2, color= "red", marker= ".", s=30, label='C_d')







# # grab a reference to the current axes
# ax = plt.gca()
# # set the xlimits to be the reverse of the current xlimits
# ax.set_xlim(ax.get_xlim()[::-1])
# # call `draw` to re-render the graph
# plt.draw()

# naming the x axis
plt.xlabel('confidence')
# naming the y axis
ax1.set_ylabel('accuracy [%] (symbols)')
ax2.set_ylabel('Number of samples (bar graph)')

plt.grid()

# show a legend on the plot
ax1.legend()

# function to show the plot
plt.show()

