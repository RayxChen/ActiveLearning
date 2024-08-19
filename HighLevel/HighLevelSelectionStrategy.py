import numpy as np
import random
import heapq
from abc import ABC, abstractmethod

# Selection Strategy Interface
class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, priority_queue):
        pass

# Different Selection Strategies
class GreedyStrategy(SelectionStrategy):
    def select(self, priority_queue):
        return heapq.heappop(priority_queue)

class EpsilonGreedyStrategy(SelectionStrategy):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def select(self, priority_queue):
        if random.uniform(0, 1) < self.epsilon:
            random_subtask = random.choice(priority_queue)
            priority_queue.remove(random_subtask)
            heapq.heapify(priority_queue)
            return random_subtask
        else:
            return heapq.heappop(priority_queue)

class ProbabilisticSamplingStrategy(SelectionStrategy):
    def select(self, priority_queue):
        total_importance = sum(subtask.importance for subtask in priority_queue)
        probabilities = [subtask.importance / total_importance for subtask in priority_queue]
        selected_subtask = np.random.choice(priority_queue, p=probabilities)
        priority_queue.remove(selected_subtask)
        heapq.heapify(priority_queue)
        return selected_subtask
