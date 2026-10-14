import matplotlib.pyplot as plt

num = []
acc = []

f = open('results/eval/hid_conf_num_dr.txt', 'r')
lines = f.readlines()
for i in range(len(lines)):
    num.append(int(lines[i]))
f.close()

f = open('results/eval/hid_conf_acc_dr.txt', 'r')
lines = f.readlines()
for i in range(len(lines)):
    acc.append(int(lines[i]))
f.close()

perc = []
for i in range(len(num)):
    perc.append(100*acc[i]/num[i])

num2 = []
acc2 = []

f = open('results/eval/hid_conf_num_hid_dr.txt', 'r')
lines = f.readlines()
for i in range(len(lines)):
    num2.append(int(lines[i]))
f.close()

f = open('results/eval/hid_conf_acc_hid_dr.txt', 'r')
lines = f.readlines()
for i in range(len(lines)):
    acc2.append(int(lines[i]))
f.close()

perc2 = []
for i in range(len(num2)):
    perc2.append(100*acc2[i]/num2[i])

x = []
for i in range(100, 49, -1):
    x.append(i)



plt.plot(x, perc, 'r-', label = 'Trained on full data (fine-tuned with dropout)')
plt.plot(x, perc2, 'g-', label = 'Trained on randomly hidden data (fine-tuned with dropout)')

# grab a reference to the current axes
ax = plt.gca()
# set the xlimits to be the reverse of the current xlimits
ax.set_xlim(ax.get_xlim()[::-1])
# call `draw` to re-render the graph
plt.draw()

# naming the x axis
plt.xlabel('Confidence')
# naming the y axis
plt.ylabel('Accuracy')

plt.grid()

# giving a title to my graph
plt.title('Accuracy on data with randomly hidden future cumulating based on confidence from softmax score')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()