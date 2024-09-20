import heapq
import numpy as np
import os
import json
from typing import List, Protocol
from Subtask import Subtask

class SelectionStrategy(Protocol):
    def select(self, queue: List[Subtask]) -> Subtask:
        ...

class ImportanceStrategy(Protocol):
    def calculate(self, subtask: Subtask) -> float:
        ...

class ActiveLearningModel:
    def __init__(self, subtasks: List[Subtask], selection_strategy: SelectionStrategy, 
                 importance_strategy: ImportanceStrategy, init_explore_size: int, 
                 loss_threshold: float, max_train_steps: int):
        
        self.subtasks = subtasks
        self.selection_strategy = selection_strategy
        self.importance_strategy = importance_strategy
        self.priority_queue: List[Subtask] = []
        self.loss_threshold = loss_threshold
        self.max_train_steps = max_train_steps
        
        # Initial training phase
        self.init_train(init_explore_size)
        for subtask in subtasks:
            heapq.heappush(self.priority_queue, subtask)

    def init_train(self, explore_size: int) -> None:
        for subtask in self.subtasks:
            subset_indices = np.random.choice(len(subtask.data), size=explore_size, replace=False)
            
            train_loss_history = subtask.trainer.train(subset_indices)
            subtask.history['train_loss'] = train_loss_history

            test_loss_history = subtask.trainer.eval()
            subtask.history['test_loss'] = test_loss_history

            # Initialize importance after initial training
            self.update_importance(subtask)

    def select_subtask(self) -> Subtask:
        return self.selection_strategy.select(self.priority_queue)

    def update_importance(self, subtask: Subtask) -> None:
        subtask.importance = self.importance_strategy.calculate(subtask)
        # Remove and re-add the subtask to update its position in the heap
        self.priority_queue = [s for s in self.priority_queue if s != subtask]
        heapq.heappush(self.priority_queue, subtask)

    def run(self) -> None:
        while not self.convergence_criteria():
            subtask = self.select_subtask()

            train_loss_history = subtask.trainer.train() 
            subtask.history['train_loss'].extend(train_loss_history) 
            test_loss_history = subtask.trainer.eval()
            subtask.history['test_loss'].extend(test_loss_history)

            self.update_importance(subtask)

    def convergence_criteria(self) -> bool:
        if not self.priority_queue:
            return True
        
        cur_task = self.priority_queue[0]
        if cur_task.history['test_loss'][-1] < self.loss_threshold:
            return True
        
        total_steps = sum(len(task.history['test_loss']) for task in self.subtasks)
        if total_steps > self.max_train_steps:
            return True
        
        return False

    def save(self, save_dir: str) -> None:
        """
        Save the ActiveLearningModel state and all subtasks.
        
        Args:
            save_dir (str): Directory to save the model and subtasks.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save ActiveLearningModel metadata
        metadata = {
            'loss_threshold': self.loss_threshold,
            'max_train_steps': self.max_train_steps,
            'selection_strategy': self.selection_strategy.__class__.__name__,
            'importance_strategy': self.importance_strategy.__class__.__name__,
        }
        with open(os.path.join(save_dir, 'active_learning_metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        # Save subtasks
        subtasks_dir = os.path.join(save_dir, 'subtasks')
        os.makedirs(subtasks_dir, exist_ok=True)
        for i, subtask in enumerate(self.subtasks):
            subtask_dir = os.path.join(subtasks_dir, f'subtask_{i}')
            subtask.save_checkpoint(subtask_dir)
        
        print(f"ActiveLearningModel saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: str, selection_strategy: SelectionStrategy, 
             importance_strategy: ImportanceStrategy) -> 'ActiveLearningModel':
        """
        Load an ActiveLearningModel from a saved state.
        
        Args:
            save_dir (str): Directory from which to load the model and subtasks.
            selection_strategy (SelectionStrategy): Strategy for selecting subtasks.
            importance_strategy (ImportanceStrategy): Strategy for calculating subtask importance.
        
        Returns:
            ActiveLearningModel: Loaded ActiveLearningModel instance.
        """
        # Load metadata
        with open(os.path.join(save_dir, 'active_learning_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Load subtasks
        subtasks_dir = os.path.join(save_dir, 'subtasks')
        subtasks = []
        for subtask_dir in sorted(os.listdir(subtasks_dir)):
            subtask_path = os.path.join(subtasks_dir, subtask_dir)
            if os.path.isdir(subtask_path):
                checkpoint_file = os.path.join(subtask_path, f"subtask_checkpoint.pth")
                subtask = Subtask.load_from_checkpoint(checkpoint_file)
                subtasks.append(subtask)
        
        # Create and return ActiveLearningModel instance
        model = cls(
            subtasks=subtasks,
            selection_strategy=selection_strategy,
            importance_strategy=importance_strategy,
            init_explore_size=0,  # We don't need to initialize again
            loss_threshold=metadata['loss_threshold'],
            max_train_steps=metadata['max_train_steps']
        )
        
        # Rebuild priority queue
        model.priority_queue = subtasks.copy()
        heapq.heapify(model.priority_queue)
        
        print(f"ActiveLearningModel loaded from {save_dir}")
        return model