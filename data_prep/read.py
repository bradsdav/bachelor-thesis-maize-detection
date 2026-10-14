import openpyxl
import os
import numpy as np
import csv
import pickle
import random

# Define variable to load the dataframe
dataframe = openpyxl.load_workbook("data/meta/dates.xlsx")

# Define variable to read sheet
dataframe1 = dataframe.active

data = []
for i in range(16):
    data.append([])

# Iterate the loop to read the cell values
for col in range(16):
    for row in dataframe1.iter_rows(0, 39):

        x = None
        if row[col].value:
            if type(row[col].value) == str:
                x = row[col].value
            elif row[col].value.month == 9:
                x = row[col].value.day
            elif row[col].value.month == 10:
                x = row[col].value.day + 30
            elif row[col].value.month == 11:
                x = row[col].value.day + 61
            elif row[col].value.month == 12:
                x = row[col].value.day + 91
            elif row[col].value.month == 1:
                x = row[col].value.day + 122
            elif row[col].value.month == 2:
                x = row[col].value.day + 153
            elif row[col].value.month == 3:
                x = row[col].value.day + 181
            elif row[col].value.month == 4:
                x = row[col].value.day + 212
            elif row[col].value.month == 5:
                x = row[col].value.day + 242
            elif row[col].value.month == 6:
                x = row[col].value.day + 273
            elif row[col].value.month == 7:
                x = row[col].value.day + 303
            elif row[col].value.month == 8:
                x = row[col].value.day + 334

        data[col].append(x)


f = open("data/meta/final.txt", "r")
final = f.read().split("\n")
final.pop()
final = list(map(int, final))


dunno = []
for i in range(16):
    dunno.append([])
    for j in range(73):
        dunno[i].append("z")


dunno.append([])
for i in range(16):
    k = 1
    dunno[16].append(data[i][0])
    for j in range(73):
        if data[i][k] and data[i][k] <= final[j]+2:
            dunno[i][j] = k-1
            k += 1


print("OK")

zero = [0]*10

data_pred = np.empty((0, 73, 10), int)
labels_pred = np.empty((0, 1), int)


data_conf = np.empty((0, 73, 10), int)
labels_conf = np.empty((0, 1), int)


base_path = "C:\\Users\\daved\\Desktop\\BC\\TimeSen2Crop\\TimeSen2Crop\\"
dir_list = os.listdir(base_path)
for dire in dunno[16]:
    if dire == "2019_33UVP" or dire == "33UVP" or dire == "33TWN":
        continue
    # if dire != "2019_33UVP" and dire != "33UVP":
    #     continue
    # if dire != "33TWN":
    #     continue
    cur = dunno[dunno[16].index(dire)]
    for k in range(16):
        if k == 2:
            label = np.array([[1]])
        else:
            label = np.array([[0]])
        # label = np.array([[k]])

        files_list = os.listdir(base_path + dire + "\\" + str(k))

        # arr_t_p = np.empty((0, 73, 10), int)
        # arr_l_p = np.empty((0, 1), int)

        hold_t_p = np.empty((0, 73, 10), int)
        hold_l_p = np.empty((0, 1), int)

        # arr_t_c = np.empty((0, 73, 10), int)
        # arr_l_c = np.empty((0, 1), int)

        hold_t_c = np.empty((0, 73, 10), int)
        hold_l_c = np.empty((0, 1), int)

        # cop = 0

        for l in files_list:
            file = open(base_path + dire + "\\" + str(k) + "\\" + l)
            csvreader = csv.reader(file)
            next(csvreader)
            mat = []
            for row in range(73):
                if cur[row] != "z":
                    h = list(map(int, next(csvreader)))
                    mat.append(h)
                else:
                    mat.append(zero)

            arr = np.array([mat])

            num = random.randint(0,99)
            if num == 0:
                hold_t_p = np.append(hold_t_p, arr, axis=0)
                hold_l_p = np.append(hold_l_p, label, axis=0)
            elif num == 1:
                hold_t_c = np.append(hold_t_c, arr, axis=0)
                hold_l_c = np.append(hold_l_c, label, axis=0)

            # cop += 1
            # if cop == 2000:
            #     cop = 0
            #     arr_t_p = np.append(arr_t_p, hold_t_p, axis=0)
            #     arr_l_p = np.append(arr_l_p, hold_l_p, axis=0)
            #     hold_t_p = np.empty((0, 73, 10), int)
            #     hold_l_p = np.empty((0, 1), int)

            #     arr_t_c = np.append(arr_t_c, hold_t_c, axis=0)
            #     arr_l_c = np.append(arr_l_c, hold_l_c, axis=0)
            #     hold_t_c = np.empty((0, 73, 10), int)
            #     hold_l_c = np.empty((0, 1), int)
        
        # arr_t_p = np.append(arr_t_p, hold_t_p, axis=0)
        # arr_l_p = np.append(arr_l_p, hold_l_p, axis=0)

        # arr_t_c = np.append(arr_t_c, hold_t_c, axis=0)
        # arr_l_c = np.append(arr_l_c, hold_l_c, axis=0)

        data_pred = np.append(data_pred, hold_t_p, axis=0)
        labels_pred = np.append(labels_pred, hold_l_p, axis=0)

        data_conf = np.append(data_conf, hold_t_c, axis=0)
        labels_conf = np.append(labels_conf, hold_l_c, axis=0)
        
        print(dire + "." + str(k))

print(labels_pred.size)
print(labels_conf.size)

pickle.dump(data_pred, open('data/confidence/train_pred_1.pkl', 'wb'))
pickle.dump(labels_pred, open('data/confidence/train_labels_pred_1.pkl', 'wb'))


pickle.dump(data_conf, open('data/confidence/train_conf_1.pkl', 'wb'))
pickle.dump(labels_conf, open('data/confidence/train_labels_conf_1.pkl', 'wb'))


# train2 = pickle.load(open('./data/rand/test.pkl', 'rb'))
# labels2 = pickle.load(open('./data/rand/test_labels.pkl', 'rb'))

print("ok")
