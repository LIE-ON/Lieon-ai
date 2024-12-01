from torch.utils.data import Dataset
import torch


class PreprocessedDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.features = data['features']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.features[idx]
        y = self.labels[idx]
        return X, y