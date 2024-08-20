import torch
import numpy as np
from utils import load_config
from Model import LitModel
import pytorch_lightning as pl
from DataHandler import DataHandler
from pytorch_lightning.loggers import TensorBoardLogger


class ActiveLearning:
    def __init__(self, strategy, model, train_dataset, test_dataset, config_file='config.yaml'):
        # Load configuration
        config = self._load_config(config_file)
        self._initialize_params(config)

        # Initialize components
        self.data = DataHandler(train_dataset, test_dataset, self.batch_size)
        self.model = self._initialize_model(model, config['training_params'])
        self.logger = self._initialize_logger(config['logging_params'])
        self.trainer = self._initialize_trainer()

        # Initialize active learning specific attributes
        self.sampled_indices = set()
        self.last_distance = None
        self.t = 0
        self.strategy = strategy
        self.dim_reducer = None

    def _load_config(self, config_file):
        """Load configuration from YAML file."""
        return load_config(config_file)

    def _initialize_params(self, config):
        """Initialize parameters from the loaded configuration."""
        training_params = config['training_params']
        active_learning_params = config['active_learning_params']

        self.tau = active_learning_params.get('tau', 0.2)
        self.decay_factor = active_learning_params.get('decay_factor', 0.95)
        self.subset_size = active_learning_params.get('subset_size', 200)
        self.max_iterations = active_learning_params.get('max_iterations', 100)
        self.cumulative = active_learning_params.get('cumulative', True)

        self.batch_size = training_params.get('batch_size', 32)
        self.epochs = training_params.get('epochs', 5)

    def _initialize_model(self, model, training_params):
        """Initialize the PyTorch Lightning model."""
        return LitModel(model, training_params)

    def _initialize_logger(self, logging_params):
        """Initialize the TensorBoard logger."""
        return TensorBoardLogger(logging_params.get('log_dir', './logs'), name="active_learning")

    def _initialize_trainer(self):
        """Initialize the PyTorch Lightning trainer."""
        return pl.Trainer(max_epochs=self.epochs, logger=self.logger, devices='auto', accelerator="gpu")

    def dimensionality_reduction(self, X):
        print('Dimensionality reduction... raw data shape:', X.shape)
        reduced_X = self.dim_reducer.fit_transform(X) if self.dim_reducer else X
        print('Dimensionality completed... reduced data shape:', reduced_X.shape)
        return reduced_X
    
    def run_sampling_strategy(self):
        # X: np.array is full train set 
        X = self.data.get_X()

        # Perform dimensionality reduction if required
        reduced_X = self.dimensionality_reduction(X)

        # Initial distribution Q in reduced space
        Q = self.strategy.get_distribution(reduced_X)

        # Separate set for Wasserstein distance computation (kept as NumPy array)
        distance_calc_set_X = np.empty((0, reduced_X.shape[1]))

        while not self.stopping_criterion():
            indices = self.strategy.sample(reduced_X, self.sampled_indices, self.subset_size)
            if len(indices) == 0:  # Traverse all the data points once
                break
            self.sampled_indices.update(indices)
            St_reduced_X = reduced_X[indices]
            # Update metric distance set as a numpy array
            distance_calc_set_X = np.vstack([distance_calc_set_X, St_reduced_X])

            # Compute distribution P and distance with Q
            P = self.strategy.get_distribution(distance_calc_set_X)
            distance = self.strategy.compute_distance(P, Q)
            print('distance', distance)

            if self.update_criterion(distance):
                if self.cumulative:
                    # Train with the cumulative dataset
                    train_dataloader = self.data.get_train_dataloader(list(self.sampled_indices))
                else:
                    train_dataloader = self.data.get_train_dataloader(indices)

                self.trainer.fit(self.model, train_dataloader)
            else:
                # Retrain with the cumulative dataset if update criterion is not met
                if len(self.sampled_indices) > 0:
                    cumulative_dataloader = self.data.get_train_dataloader(list(self.sampled_indices))
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
        return self.t >= self.max_iterations or len(self.sampled_indices) >= len(self.data.train_dataset)
    
    def test_performance(self):
        test_dataloader = self.data.get_test_dataloader()
        self.trainer.test(self.model, dataloaders=test_dataloader)

    def add_new_data(self, new_train_data=None, new_test_data=None):
        # TODO: when to add the new data in the AL process
        # need to know the data collection speed, training speed, etc.

        """Add new data to the existing datasets."""
        if new_train_data:
            self.data.update_train_dataset(new_train_data)
        if new_test_data:
            self.data.update_test_dataset(new_test_data)
