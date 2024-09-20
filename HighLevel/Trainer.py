from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

# Local imports
from Subtask import SubtaskData
from utils import set_random_state


class Trainer:
    def __init__(self, global_params: Dict[str, Any], subtask_params: Dict[str, Any], log_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module = subtask_params['model']
        self.dataset: SubtaskData = subtask_params['dataset']
        
        self.batch_size: int = subtask_params.get('batch_size', global_params['trainer']['batch_size'])
        self.learning_rate: float = subtask_params.get('learning_rate', global_params['trainer']['learning_rate'])
        self.epochs: int = subtask_params.get('epochs', global_params['trainer']['epochs'])
        self.test_size: float = subtask_params.get('test_size', global_params['trainer']['test_size'])
        
        self.random_seed: int = global_params['random_seed']
        set_random_state(self.random_seed)

        self.train_dataset, self.test_dataset = self.train_test_split_dataset(
            self.dataset, self.test_size
        )

        self.update_dataloaders()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_test_split_dataset(
        self,
        dataset: SubtaskData,
        test_size: float
    ) -> Tuple[Subset, Subset]:
        train_indices, test_indices = train_test_split(
            list(dataset.active_indices),
            test_size=test_size,
            random_state=self.random_seed
        )
        return (
            Subset(dataset, train_indices),
            Subset(dataset, test_indices)
        )

    def update_dataloaders(self) -> None:
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self, subset_indices: Optional[List[int]] = None) -> List[float]:
        train_loss_history = []
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            
            if subset_indices is None:
                train_dataloader = self.train_dataloader
            else:
                subset = Subset(self.train_dataset, subset_indices)
                train_dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)

            for batch_idx, (inputs, labels) in enumerate(train_dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                global_step = epoch * len(train_dataloader) + batch_idx
                self.writer.add_scalar('Training Loss (Batch)', loss.item(), global_step)
                train_loss_history.append(loss.item())

                running_loss += loss.item()

            avg_loss = running_loss / len(train_dataloader)
            self.writer.add_scalar('Average Training Loss (Epoch)', avg_loss, epoch)

        self.writer.close()
        self.model.to("cpu")
        torch.cuda.empty_cache()

        return train_loss_history

    def eval(self) -> List[float]:
        self.model.to(self.device)
        test_loss_history = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss_history.append(loss.item())

        return test_loss_history