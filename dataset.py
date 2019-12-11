import pickle
from torch.utils.data import Dataset
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, root_dir, file_name, train=True):
        f = open(os.path.join(root_dir, file_name), "rb")
        dataset = pickle.load(f)
        if train:
            self.data = dataset['tr_data']
            self.label = dataset['tr_lbl']
        else:
            self.data = dataset['te_data']
            self.label = dataset['te_lbl']
        self.len = self.label.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
