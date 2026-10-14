import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class myDataset(Dataset):
    def __init__(self, data_p, lab_p):
        self.data = pickle.load(open(data_p, 'rb'))
        self.labels = pickle.load(open(lab_p, 'rb'))
        self.data = torch.from_numpy(self.data)
        self.labels = torch.from_numpy(self.labels)
        self.size = self.data.shape[0]

    def __getitem__(self, index):
        # stored as (73 time steps, 10 bands) -> transpose so bands are the conv channels
        dat = torch.transpose(self.data[index], 0, 1)
        dat = dat.to(torch.float32)
        
        # label = self.labels[index]
        # label = label.to(torch.int64)

        label = torch.zeros(2)
        label[int(self.labels[index])] = 1
        label = label.to(torch.float32)

        return {'data': dat, 'label': label}

    def __len__(self):
        return self.size

