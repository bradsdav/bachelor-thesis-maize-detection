import matplotlib.pyplot as plt

# fig, ax1 = plt.subplots()

# ax2 = ax1.twinx()

acc1 = []
loss1 = []

f = open('results/eval/test_baseline.txt', 'r')
lines = f.readlines()
for i in range(len(lines)):
    if i % 2 == 1:
        acc1.append(float(lines[i]))
    else:
        loss1.append(float(lines[i]))
f.close()

acc2 = []
loss2 = []

f = open('results/eval/test_robust.txt', 'r')
lines = f.readlines()
for i in range(len(lines)):
    if i % 2 == 1:
        acc2.append(float(lines[i]))
    else:
        loss2.append(float(lines[i]))
f.close()

acc3 = []
loss3 = []

f = open('results/eval/test_spec.txt', 'r')
lines = f.readlines()
for i in range(len(lines)):
    # if i % 2 == 1:
    acc3.append(float(lines[i]))
    # else:
    #     loss3.append(float(lines[i]))
f.close()

x = []
for i in range(12):
    x.append(100 - 100*(((11-i)*7)/73))
x[0] = 0

x.reverse()

# y1 = [88.025,88.65,89.2,84.773,84.4,83.184,86.221,86.977,90.717,86.768,86.467,86.126,86.079,86.091,85.613]
# y2 = [0.349,0.340,0.321,0.349,0.340,0.321,0.349,0.340,0.321,0.349,0.340,0.321,0.349,0.340,0.321]

# ax1.plot(x, y1, 'r-')
# ax2.plot(x, y2, 'g-')


# ax1.set_xlabel('Sparsity [%]')
# ax1.set_ylabel('Accuracy', color='r')
# ax2.set_ylabel('Loss', color='g')


plt.plot(x, acc1, 'r-', label = 'baseline model')
plt.plot(x, acc2, 'g-', label = 'robust model')
plt.plot(x, acc3, 'b-', label = 'specialised models')

# # grab a reference to the current axes
# ax = plt.gca()
# # set the xlimits to be the reverse of the current xlimits
# ax.set_xlim(ax.get_xlim()[::-1])
# # call `draw` to re-render the graph
# plt.draw()

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
