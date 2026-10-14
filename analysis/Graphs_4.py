import matplotlib.pyplot as plt

x = []
for i in range(100, 49, -1):
    x.append(i)

for dd in range(11):
    # if dd % 2 == 1:
    #     continue

    num = []
    acc = []

    f = open('results/eval/conf_num_'+str(dd)+'.txt', 'r')
    lines = f.readlines()
    for i in range(len(lines)):
        num.append(int(lines[i]))
    f.close()

    f = open('results/eval/conf_acc_'+str(dd)+'.txt', 'r')
    lines = f.readlines()
    for i in range(len(lines)):
        acc.append(int(lines[i]))
    f.close()

    perc = []
    for i in range(len(num)):
        perc.append(100*acc[i]/num[i])

    plt.plot(x, perc, label = str(dd))



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
plt.title('Accuracy on full data cumulating based on confidence from softmax score')

# show a legend on the plot
plt.legend()

# function to show the plot
plt.show()
