from abc import ABC, abstractmethod
from typing import List, Literal, TypedDict, TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy.stats import gmean

if TYPE_CHECKING:
    from Subtask import Subtask

class LossHistory(TypedDict):
    train_loss: List[float]
    test_loss: List[float]

class StatsDict(TypedDict):
    pct_change: np.ndarray
    mean: float
    std: float

class StatisticsDict(TypedDict):
    train_stats: StatsDict
    test_stats: StatsDict

# Importance Calculation Strategy Interface
class ImportanceCalculationStrategy(ABC):
    @abstractmethod
    def calculate(self, subtask: 'Subtask') -> float:
        pass

    def calculate_statistics(self, subtask: 'Subtask', mean_type: Literal['arithmetic', 'geometric'] = 'geometric') -> StatisticsDict:
        train_loss_history = subtask.history['train_loss']
        test_loss_history = subtask.history['test_loss']

        def calculate_stats(loss_history: List[float], mean_type: Literal['arithmetic', 'geometric']) -> StatsDict:
            array = np.array(loss_history)
            pct_change = np.diff(array) / array[:-1]
            
            if mean_type == 'arithmetic':
                mu = np.mean(pct_change)
                sigma = np.std(pct_change)
                return {"pct_change": pct_change, "mean": mu, "std": sigma}
            elif mean_type == 'geometric':
                geometric_mean = gmean(1 + pct_change) - 1
                log_pct_change = np.log(1 + pct_change)
                geometric_std = np.std(log_pct_change)
                return {"pct_change": log_pct_change, "mean": geometric_mean, "std": geometric_std}
            else:
                raise ValueError("Invalid mean_type. Choose either 'arithmetic' or 'geometric'.")

        train_stats = calculate_stats(train_loss_history, mean_type)
        test_stats = calculate_stats(test_loss_history, mean_type)

        return {"train_stats": train_stats, "test_stats": test_stats}
    
    def apply_time_weight(self, importance: float, sampling_time: int) -> float:
        return importance * (1 + self.time_weight * np.log(sampling_time + 1))

# Different Importance Calculation Strategies
class EMAStrategy(ImportanceCalculationStrategy):
    def __init__(self, alpha: float = 0.8, lambda_: float = 1.0, mu: float = 1.0, time_weight: float = 0.0):
        super().__init__(time_weight)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.mu = mu

    def calculate(self, subtask: 'Subtask') -> float:
        data_dict = self.calculate_statistics(subtask)
        train_pct_change = data_dict['train_stats']['pct_change']
        test_pct_change = data_dict['test_stats']['pct_change']

        train_ema = self._calculate_ema(train_pct_change)
        test_ema = self._calculate_ema(test_pct_change)

        importance = 0.0
        if train_ema is not None:
            importance += self.lambda_ * abs(train_ema)
        if test_ema is not None:
            importance += self.mu * abs(test_ema)

        return self.apply_time_weight(importance, subtask.sampling_time)

    def _calculate_ema(self, pct_change: np.ndarray) -> float:
        ema = pd.Series(pct_change).ewm(alpha=self.alpha, adjust=False).mean().iloc[-1]
        return ema

class MaxMeanStrategy(ImportanceCalculationStrategy):
    def __init__(self, mean_type: Literal['arithmetic', 'geometric'] = 'geometric', lambda_: float = 1.0, mu: float = 1.0, time_weight: float = 0.0):
        super().__init__(time_weight)
        self.mean_type = mean_type
        self.lambda_ = lambda_
        self.mu = mu

    def calculate(self, subtask: 'Subtask') -> float:
        data_dict = self.calculate_statistics(subtask, mean_type=self.mean_type)
        train_mean = data_dict['train_stats']['mean']
        test_mean = data_dict['test_stats']['mean']

        importance = (self.lambda_ * train_mean) + (self.mu * test_mean)

        return self.apply_time_weight(importance, subtask.sampling_time)

class MaxStdStrategy(ImportanceCalculationStrategy):
    def __init__(self, mean_type: Literal['arithmetic', 'geometric'] = 'geometric', lambda_: float = 1.0, mu: float = 1.0, time_weight: float = 0.0):
        super().__init__(time_weight)
        self.mean_type = mean_type
        self.lambda_ = lambda_
        self.mu = mu

    def calculate(self, subtask: 'Subtask') -> float:
        data_dict = self.calculate_statistics(subtask, mean_type=self.mean_type)
        train_std = data_dict['train_stats']['std']
        test_std = data_dict['test_stats']['std']

        importance = (self.lambda_ * train_std) + (self.mu * test_std)

        return self.apply_time_weight(importance, subtask.sampling_time)

class MaxRatioStrategy(ImportanceCalculationStrategy):
    def __init__(self, mean_type: Literal['arithmetic', 'geometric'] = 'geometric', lambda_: float = 1.0, mu: float = 1.0, time_weight: float = 0.0):
        super().__init__(time_weight)
        self.mean_type = mean_type
        self.lambda_ = lambda_
        self.mu = mu

    def calculate(self, subtask: 'Subtask') -> float:
        data_dict = self.calculate_statistics(subtask, mean_type=self.mean_type)
        train_mean = data_dict['train_stats']['mean']
        train_std = data_dict['train_stats']['std']
        test_mean = data_dict['test_stats']['mean']
        test_std = data_dict['test_stats']['std']

        train_ratio = self._calculate_ratio(train_mean, train_std)
        test_ratio = self._calculate_ratio(test_mean, test_std)

        importance = (self.lambda_ * train_ratio) + (self.mu * test_ratio)

        return self.apply_time_weight(importance, subtask.sampling_time)

    def _calculate_ratio(self, mean: float, std: float) -> float:
        return mean / (std + 1e-10)

class MaxProductStrategy(ImportanceCalculationStrategy):
    def __init__(self, mean_type: Literal['arithmetic', 'geometric'] = 'geometric', time_weight: float = 0.0):
        super().__init__(time_weight)
        self.mean_type = mean_type

    def calculate(self, subtask: 'Subtask') -> float:
        data_dict = self.calculate_statistics(subtask, mean_type=self.mean_type)
        train_mean = data_dict['train_stats']['mean']
        train_std = data_dict['train_stats']['std']
        test_mean = data_dict['test_stats']['mean']
        test_std = data_dict['test_stats']['std']

        importance = train_mean * train_std * test_mean * test_std

        return self.apply_time_weight(importance, subtask.sampling_time)