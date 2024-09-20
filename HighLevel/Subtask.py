import os
import yaml
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Dict, Any, Set, Tuple
from Trainer import Trainer

class SubtaskData(Dataset):
    def __init__(self, data: pd.DataFrame, active_indices: Set[int], is_dynamic: bool):
        self.data = data
        self.active_indices = active_indices
        self.is_dynamic = is_dynamic

    def __len__(self) -> int:
        return len(self.active_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self):
            raise IndexError("Index out of range")
        
        active_idx = list(self.active_indices)[idx]
        row = self.data.iloc[active_idx]
        
        # Assuming the data is properly structured with separate feature and target columns
        features = torch.tensor(row[self.feature_columns].values, dtype=torch.float32)
        target = torch.tensor(row[self.target_column], dtype=torch.float32)
        
        return features, target

    def add_samples(self, new_samples: pd.DataFrame):
        if self.is_dynamic:
            start_index = len(self.data)
            self.data = pd.concat([self.data, new_samples], ignore_index=True)
            self.active_indices.update(range(start_index, len(self.data)))
        else:
            print("Warning: Attempting to add samples to a static dataset. No changes made.")

class Subtask:
    def __init__(self, model: nn.Module, dataset: SubtaskData, trainer: Trainer, 
                 train_params: Dict[str, Any], initial_importance: float, sampling_time: int):
        self.model = model
        self.data = dataset 
        self.trainer = trainer
        self.train_params = train_params
        self.importance = initial_importance
        self.sampling_time = sampling_time
        self.history: Dict[str, List[float]] = {'train_loss': [], 'test_loss': []}

    def __lt__(self, other: 'Subtask') -> bool:
        return self.importance > other.importance  # min heap

    def update_dataset(self, new_samples: pd.DataFrame) -> None:
        if self.data.is_dynamic:
            self.data.add_samples(new_samples)
            self.trainer.update_dataloaders()
        else:
            print("Warning: Attempting to update a static dataset. No changes made.")

    @staticmethod
    def load_model(model_path: str) -> nn.Module:
        return torch.load(model_path) # load from dict perhaps

    @classmethod
    def build_subtasks_from_config(cls, config_path: str) -> List['Subtask']:
        """
        Factory method to create multiple Subtask instances from a configuration file.

        This method acts as a factory, constructing multiple Subtask objects by interpreting
        the provided configuration file. It handles the creation and setup of models,
        datasets, and trainers for each subtask defined in the configuration.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            List[Subtask]: A list of fully configured Subtask instances.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        global_config = config['global']
        subtasks = []

        for subtask_config in config['subtasks']:
            # Load the model
            model = cls.load_model(subtask_config['model_path'])

            # Prepare the dataset
            initial_data = pd.read_csv(subtask_config['data_path'])
            is_dynamic = subtask_config.get('is_dynamic', False)
            dataset = SubtaskData(initial_data, set(range(len(initial_data))), is_dynamic)

            # Prepare trainer parameters
            trainer_params = {**global_config['trainer'], **subtask_config.get('trainer', {})}
            trainer_params['model'] = model
            trainer_params['dataset'] = dataset

            # Create the trainer
            trainer = Trainer(global_config, trainer_params, log_dir=subtask_config['log_dir'])

            # Construct the Subtask instance
            subtask = cls(
                model=model,
                dataset=dataset,
                trainer=trainer,
                train_params=trainer_params,
                initial_importance=subtask_config['initial_importance'],
                sampling_time=subtask_config['sampling_time']
            )
            subtasks.append(subtask)

        return subtasks

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"subtask_{id(self)}_checkpoint.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'importance': self.importance,
            'sampling_time': self.sampling_time,
            'history': self.history,
            'train_params': self.train_params,
            'data': self.data.data,
            'active_indices': self.data.active_indices,
            'is_dynamic': self.data.is_dynamic
        }, checkpoint_path)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, model: nn.Module, global_config: Dict[str, Any]) -> 'Subtask':
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        dataset = SubtaskData(checkpoint['data'], checkpoint['active_indices'], checkpoint['is_dynamic'])
        
        trainer_params = checkpoint['train_params']
        trainer_params['model'] = model
        trainer_params['dataset'] = dataset

        trainer = Trainer(global_config, trainer_params, log_dir='logs')
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        subtask = cls(
            model=model,
            dataset=dataset,
            trainer=trainer,
            train_params=trainer_params,
            initial_importance=checkpoint['importance'],
            sampling_time=checkpoint['sampling_time']
        )
        subtask.history = checkpoint['history']
        
        return subtask
