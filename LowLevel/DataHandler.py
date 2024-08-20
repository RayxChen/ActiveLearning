import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset


class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        """
        Custom Dataset for loading data from NumPy arrays.

        Args:
            X (np.array): The data features as a NumPy array.
            y (np.array): The labels/targets as a NumPy array.
            transform (callable, optional): Optional transform to be applied on the data. Defaults to converting to tensor and standard normal.
            target_transform (callable, optional): Optional transform to be applied on the targets. Defaults to standard normal.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Data (X) and targets (y) must have the same number of samples")
        
        self.X = X
        self.y = y
        
        if transform is None:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)   
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transform

        if target_transform is None:
            y_mean = np.mean(y)
            y_std = np.std(y)
            self.target_transform = transforms.Compose([
                transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.float32)),
                transforms.Normalize(mean=[y_mean], std=[y_std])
            ])
        else:
            self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            X = self.transform(X)
        
        if self.target_transform:
            y = self.target_transform(y)
        
        return X, y


class DataHandler:
    """
    DataHandler class to manage training and testing datasets and their data loaders.

    Args:
        train_dataset (Dataset): The training dataset, should be an instance of a class that inherits from torch.utils.data.Dataset.
        test_dataset (Dataset): The testing dataset, should be an instance of a class that inherits from torch.utils.data.Dataset.
        batch_size (int): The batch size for data loaders.
        shuffle (bool): Whether to shuffle the data during training. Defaults to True.
    """
    def __init__(self, train_dataset, test_dataset, batch_size, shuffle=True):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_dataloader = None
        self.test_dataloader = None

        self._initialize_dataloaders()
        self.X = self._initialize_X()
    
    def _initialize_X(self):        
        return self.train_dataset.data # np array
    
    def _initialize_dataloaders(self):
        """Initialize or refresh the data loaders."""
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_train_subset(self, indices):
        """Return a subset of the training dataset based on the provided indices."""
        return Subset(self.train_dataset, indices)

    def get_train_dataloader(self, indices=None):
        """Get the training data loader. If indices are provided, return a loader for the subset."""
        if indices is not None:
            subset = self.get_train_subset(indices)
            return DataLoader(subset, batch_size=self.batch_size, shuffle=self.shuffle)
        return self.train_dataloader

    def get_test_dataloader(self):
        """Get the test data loader."""
        return self.test_dataloader

    def update_train_dataset(self, new_data):
        """Append new data to the existing training dataset and refresh the data loader."""

        # Concatenate the underlying NumPy arrays
        self.X = np.concatenate([self.X, new_data.data])

        self.train_dataset = ConcatDataset([self.train_dataset, new_data])
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def update_test_dataset(self, new_data):
        """Append new data to the existing test dataset and refresh the data loader."""
        self.test_dataset = ConcatDataset([self.test_dataset, new_data])
        self.test_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def get_X(self):
        return self.X
    