import torch
import numpy as np
from utils import load_config
from Model import LitModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.loggers import TensorBoardLogger


class ActiveLearning:
    def __init__(
        self,
        strategy,
        model,
        train_dataset,
        test_dataset,
        config_file='config.yaml'
    ):
        # Load configuration from YAML file
        config = load_config(config_file)
        training_params = config['training_params']
        active_learning_params = config['active_learning_params']
        logging_params = config['logging_params']

        self.tau = active_learning_params.get('tau', 0.2)
        self.decay_factor = active_learning_params.get('decay_factor', 0.95)
        self.subset_size = active_learning_params.get('subset_size', 200)
        self.max_iterations = active_learning_params.get('max_iterations', 100)
        self.cumulative = active_learning_params.get('cumulative', True)
        
        self.batch_size = training_params.get('batch_size', 32)
        self.epochs = training_params.get('epochs', 5)

        self.strategy = strategy
        self.dim_reducer = None

        self.sampled_indices = set()
        self.last_distance = None 
        self.t = 0

        # Initialize PyTorch Lightning components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LitModel(model, training_params).to(self.device) 
        
        self.train_dataset = train_dataset
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.logger = TensorBoardLogger(logging_params.get('log_dir', './logs'), name="active_learning")
        self.trainer = pl.Trainer(max_epochs=self.epochs, logger=self.logger, devices='auto', accelerator="gpu")

    def dimensionality_reduction(self, X):
        print('Dimensionality reduction... raw data shape:', X.shape)
        reduced_X = self.dim_reducer.fit_transform(X) if self.dim_reducer else X
        print('Dimensionality completed... reduced data shape:', reduced_X.shape)
        return reduced_X
    
    def run_sampling_strategy(self, X):
        # X is full train set 

        # Perform dimensionality reduction if required
        reduced_X = self.dimensionality_reduction(X)

        # Initial distribution Q in reduced space
        Q = self.strategy.get_distribution(reduced_X)

        # Separate set for Wasserstein distance computation (kept as NumPy array)
        distance_calc_set_X = np.empty((0, reduced_X.shape[1]))

        while not self.stopping_criterion():
            indices = self.strategy.sample(reduced_X, self.sampled_indices, self.subset_size)
            if len(indices) == 0: # traverse all the data points for once
                break
            self.sampled_indices.update(indices)
            St_reduced_X = reduced_X[indices]
            # Update metric distance set as a numpy array
            distance_calc_set_X = np.vstack([distance_calc_set_X, St_reduced_X])

            # Compute distribution P and Wasserstein distance
            P = self.strategy.get_distribution(distance_calc_set_X)
            distance = self.strategy.compute_distance(P, Q)
            print('distance', distance)

            if self.update_criterion(distance):
                if self.cumulative:
                    # Train with the cumulative dataset
                    train_subset = Subset(self.train_dataset, list(self.sampled_indices))
                else:
                    train_subset = Subset(self.train_dataset, indices)

                train_dataloader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=False)
                self.trainer.fit(self.model, train_dataloader)
            else:
                # Retrain with the cumulative dataset if update criterion is not met
                if len(self.sampled_indices) > 0:
                    cumulative_dataset = Subset(self.train_dataset, list(self.sampled_indices))
                    cumulative_dataloader = DataLoader(cumulative_dataset, batch_size=self.batch_size, shuffle=False)
                    self.trainer.fit(self.model, cumulative_dataloader)

            self.last_distance = distance
            self.tau *= self.decay_factor
            self.t += 1

        return self.model
    
    def update_criterion(self, distance):
        if self.last_distance is None or distance > self.last_distance:
            return False
        diff_ratio = (self.last_distance - distance) / self.last_distance
        return diff_ratio <= self.tau  # distributions within a pct threshold tau

    def stopping_criterion(self):
        return self.t >= self.max_iterations or len(self.sampled_indices) >= len(self.train_dataset)
    

    def test_performance(self):
        self.trainer.test(self.model, dataloaders=self.test_dataloader)