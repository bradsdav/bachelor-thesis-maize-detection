import pickle

train = pickle.load(open('./data/orig/train_16.pkl', 'rb'))
tr_labels = pickle.load(open('./data/orig/train_labels_16.pkl', 'rb'))

validation = pickle.load(open('./data/orig/validation_16.pkl', 'rb'))
val_labels = pickle.load(open('./data/orig/validation_labels_16.pkl', 'rb'))

test = pickle.load(open('./data/orig/test_16.pkl', 'rb'))
test_labels = pickle.load(open('./data/orig/test_labels_16.pkl', 'rb'))

train_dat = dict()
val_dat = dict()
test_dat = dict()

train_len = 0
val_len = 0
test_len = 0

for i in range(16):
    train_dat[str(i)] = 0
    val_dat[str(i)] = 0
    test_dat[str(i)] = 0

for label in tr_labels:
    train_dat[str(label[0])] += 1
    train_len += 1

for label in val_labels:
    val_dat[str(label[0])] += 1
    val_len += 1

for label in test_labels:
    test_dat[str(label[0])] += 1
    test_len += 1


print(train_dat)
print(val_dat)
print(test_dat)

for x in train_dat:
    train_dat[x] = 100 * train_dat[x]/train_len

for x in val_dat:
    val_dat[x] = 100 * val_dat[x]/val_len

for x in test_dat:
    test_dat[x] = 100 * test_dat[x]/test_len

print(train_dat)
print(val_dat)
print(test_dat)
