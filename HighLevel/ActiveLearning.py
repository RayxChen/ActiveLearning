import heapq
import numpy as np


class ActiveLearningModel:
    def __init__(self, subtasks, selection_strategy, importance_strategy,
                 init_explore_size, loss_threshold, max_train_steps):
        
        self.subtasks = subtasks
        self.selection_strategy = selection_strategy
        self.importance_strategy = importance_strategy
        self.priority_queue = []
        self.loss_threshold = loss_threshold
        self.max_train_steps = max_train_steps
        
        # Initial training phase
        self.init_train(init_explore_size) # init data points exploration/cold start        
        for subtask in subtasks:
            heapq.heappush(self.priority_queue, subtask)

    def init_train(self, explore_size):
        for subtask in self.subtasks:
            subset_indices = np.random.choice(len(subtask.data), size=explore_size, replace=False)
            
            train_loss_history = subtask.trainer.train(subset_indices)
            subtask.history['train_loss'] = train_loss_history

            test_loss_history = subtask.trainer.eval()
            subtask.history['test_loss'] = test_loss_history


    def select_subtask(self):
        return self.selection_strategy.select(self.priority_queue)

    def update_importance(self, subtask):
        subtask.importance = self.importance_strategy.calculate(subtask)
        heapq.heappush(self.priority_queue, subtask)

    def run(self):
        while not self.convergence_criteria():
            subtask = self.select_subtask()

            train_loss_history = subtask.trainer.train() 
            subtask.history['train_loss'].extend(train_loss_history) 
            test_loss_history = subtask.trainer.eval()
            subtask.history['test_loss'].extend(test_loss_history)

            self.update_importance(subtask)

    def convergence_criteria(self):
        cur_task = self.priority_queue[0]
        if cur_task.history['test_loss'] < self.loss_threshold:
            return True
        
        total_steps = sum([len(task.history['test_loss']) for task in self.subtasks])
        if total_steps > self.max_train_steps:
            return True
        




