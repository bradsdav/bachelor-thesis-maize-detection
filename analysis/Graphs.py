import matplotlib.pyplot as plt

y1 = []
y2 = []

f = open('results/logs/accuracy_robust.txt', 'r')
lines = f.readlines()
for i in range(len(lines)):
    if i % 2 == 1:
        y1.append(float(lines[i]))
    else:
        y2.append(float(lines[i]))
f.close()

x = []
for i in range(len(y1)):
    x.append(i+1)

# y1 = [88.025,88.65,89.2,84.773,84.4,83.184,86.221,86.977,90.717,86.768,86.467,86.126,86.079,86.091,85.613]
# y2 = [0.349,0.340,0.321,0.349,0.340,0.321,0.349,0.340,0.321,0.349,0.340,0.321,0.349,0.340,0.321]

plt.plot(x, y1, 'r-', label = 'Validation data')
plt.plot(x, y2, 'g-', label = 'Training data')

# naming the x axis
plt.xlabel('Epoch')
# naming the y axis
plt.ylabel('accuracy [%]')

plt.grid()

# giving a title to my graph
# plt.title('baseline')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()
