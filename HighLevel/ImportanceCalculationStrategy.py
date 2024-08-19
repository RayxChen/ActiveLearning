import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import gmean

# Importance Calculation Strategy Interface
class ImportanceCalculationStrategy(ABC):
    @abstractmethod
    def calculate(self, subtask, model):
        pass

    def calculate_statistics(self, subtask, return_type='geometric'):
        train_loss_history = subtask.history['train_loss']
        test_loss_history = subtask.history['test_loss']

        def calculate_stats(loss_history, return_type):
            array = np.array(loss_history)

            # Calculate percentage changes
            pct_change = np.diff(array) / array[:-1]
            
            if return_type == 'arithmetic':
                # Calculate arithmetic mean and standard deviation
                mu = np.mean(pct_change)
                sigma = np.std(pct_change)
                return {
                    "pct_change": pct_change,
                    "mean": mu,
                    "std": sigma
                }
            elif return_type == 'geometric':
                # Calculate geometric mean and standard deviation
                geometric_mean = gmean(1 + pct_change) - 1  # Adjust for percentage change
                log_pct_change = np.log(1 + pct_change)
                geometric_std = np.std(log_pct_change)
                return {
                    "pct_change": log_pct_change,
                    "mean": geometric_mean,
                    "std": geometric_std
                }
            else:
                raise ValueError("Invalid return_type. Choose either 'arithmetic' or 'geometric'.")

        # Calculate statistics for both train and test loss histories based on the flag
        train_stats = calculate_stats(train_loss_history, return_type)
        test_stats = calculate_stats(test_loss_history, return_type)

        # Return the statistics as a dictionary
        return {
            "train_stats": train_stats,
            "test_stats": test_stats
        }
    

        
# Different Importance Calculation Strategies
class EMAStrategy(ImportanceCalculationStrategy):
    def __init__(self, alpha=0.8, lambda_=1.0, mu=1.0):
        """
        :param alpha: Decay factor for the EMA.
        :param lambda_: Weight for the train loss EMA.
        :param mu: Weight for the test loss EMA.
        """
        self.alpha = alpha  # Decay factor for the EMA
        self.lambda_ = lambda_  # Weight for train loss EMA
        self.mu = mu  # Weight for test loss EMA

    def calculate(self, subtask):
        data_dict = self.calculate_statistics(subtask)

        # Extract the percentage changes for train and test losses
        train_pct_change = data_dict['train_stats']['pct_change']
        test_pct_change = data_dict['test_stats']['pct_change']

        # Calculate the EMA for the percentage changes
        train_ema = self._calculate_ema(train_pct_change)
        test_ema = self._calculate_ema(test_pct_change)

        # Calculate importance using weighted sum of train and test EMA
        importance = 0
        if train_ema is not None:
            importance += self.lambda_ * abs(train_ema)
        if test_ema is not None:
            importance += self.mu * abs(test_ema)

        return importance

    def _calculate_ema(self, pct_change):
        if len(pct_change) > 0:
            # Compute the EMA using pandas with alpha
            ema = pd.Series(pct_change).ewm(alpha=self.alpha, adjust=False).mean().iloc[-1]
            return ema
        else:
            return None

class MaxMeanStrategy(ImportanceCalculationStrategy):
    def __init__(self, return_type='geometric', lambda_=1.0, mu=1.0):
        """
        :param return_type: Type of mean to calculate ('arithmetic' or 'geometric').
        :param lambda_: Weight for the train mean.
        :param mu: Weight for the test mean.
        """
        self.return_type = return_type
        self.lambda_ = lambda_  # Weight for train standard deviation
        self.mu = mu  # Weight for test standard deviation

    def calculate(self, subtask):
        data_dict = self.calculate_statistics(subtask, return_type=self.return_type)

        # Extract mean and standard deviation for train and test
        train_mean = data_dict['train_stats']['mean']
        test_mean = data_dict['test_stats']['mean']

        # Calculate importance based on maximizing the mean
        importance = (self.lambda_ * train_mean) + (self.mu * test_mean)

        return importance


class MaxStdStrategy(ImportanceCalculationStrategy):
    def __init__(self, return_type='geometric', lambda_=1.0, mu=1.0):
        """
        :param return_type: Type of mean to calculate ('arithmetic' or 'geometric').
        :param lambda_: Weight for the train standard deviation.
        :param mu: Weight for the test standard deviation.
        """
        self.return_type = return_type
        self.lambda_ = lambda_  # Weight for train standard deviation
        self.mu = mu  # Weight for test standard deviation

    def calculate(self, subtask):
        data_dict = self.calculate_statistics(subtask, return_type=self.return_type)

        train_std = data_dict['train_stats']['std']
        test_std = data_dict['test_stats']['std']

        # Calculate importance based on maximizing the standard deviation
        importance = (self.lambda_ * train_std) + (self.mu * test_std)

        return importance
    
        
class MaxRatioStrategy(ImportanceCalculationStrategy):
    def __init__(self, return_type='geometric', lambda_=1.0, mu=1.0):
  
        self.return_type = return_type
        self.lambda_ = lambda_  # Weight for train ratio
        self.mu = mu  # Weight for test ratio

    def calculate(self, subtask):
        data_dict = self.calculate_statistics(subtask, return_type=self.return_type)

        # Extract mean and standard deviation for train and test
        train_mean = data_dict['train_stats']['mean']
        train_std = data_dict['train_stats']['std']
        test_mean = data_dict['test_stats']['mean']
        test_std = data_dict['test_stats']['std']

        train_ratio = self._calculate_ratio(train_mean, train_std)
        test_ratio = self._calculate_ratio(test_mean, test_std)

        # Combine the ratios for train and test using lambda_ and mu weights
        importance = (self.lambda_ * train_ratio) + (self.mu * test_ratio)

        return importance

    def _calculate_ratio(self, mean, std):
        if std != 0:
            return mean / std
        else:
            return float('inf') 
    

class MaxProductStrategy(ImportanceCalculationStrategy):
    def __init__(self, return_type='geometric'):
        self.return_type = return_type

    def calculate(self, subtask):
        data_dict = self.calculate_statistics(subtask, return_type=self.return_type)

        train_mean = data_dict['train_stats']['mean']
        train_std = data_dict['train_stats']['std']
        test_mean = data_dict['test_stats']['mean']
        test_std = data_dict['test_stats']['std']

        importance = train_mean * train_std * test_mean * test_std

        return importance