import pickle

train = pickle.load(open('./data/orig/train_or.pkl', 'rb'))
tr_labels = pickle.load(open('./data/orig/train_labels_or.pkl', 'rb'))

validation = pickle.load(open('./data/orig/validation_or.pkl', 'rb'))
val_labels = pickle.load(open('./data/orig/validation_labels_or.pkl', 'rb'))

test = pickle.load(open('./data/orig/test_or.pkl', 'rb'))
test_labels = pickle.load(open('./data/orig/test_labels_or.pkl', 'rb'))

train_maize = 0
val_maize = 0
test_maize = 0

for label in tr_labels:
    if label == 1:
        train_maize += 1

for label in val_labels:
    if label == 1:
        val_maize += 1

for label in test_labels:
    if label == 1:
        test_maize += 1

print('train maize:', train_maize)
print('complete training:', len(tr_labels))
print('Training maize percentage:', 100*train_maize/len(tr_labels), '%')
print('Training Non-maize percentage:', 100 - 100*train_maize/len(tr_labels), '%')
print('\n')

print('validation maize:', val_maize)
print('complete validation:', len(val_labels))
print('validation maize percentage:', 100*val_maize/len(val_labels), '%')
print('validation Non-maize percentage:', 100 - 100*val_maize/len(val_labels), '%')
print('\n')

print('testing maize:', test_maize)
print('complete testing:', len(test_labels))
print('testing maize percentage:', 100*test_maize/len(test_labels), '%')
print('testing Non-maize percentage:', 100 - 100*test_maize/len(test_labels), '%')
print('\n')
