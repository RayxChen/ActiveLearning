import numpy as np
from SamplingStrategy import SamplingStrategy


class ActiveLearning:
    def __init__(
        self,
        strategy: SamplingStrategy,
        tau = 0.2, 
        decay_factor=0.9,
        subset_size=200,
        max_iterations=100,
        dim_reducer=None,
    ):  
        self.strategy = strategy
        self.tau = tau
        self.decay_factor = decay_factor
        self.subset_size = subset_size
        self.max_iterations = max_iterations
        self.dim_reducer = dim_reducer
        
        self.sampled_indices = set()
        self.Scumulative_X = None
        self.Scumulative_y = None
        self.X_size = None
        self.last_distance = None 
        self.t = 0

    def dimensionality_reduction(self, X):
        print('Dimensionality reduction... raw data shape:', X.shape)
        reduced_X = self.dim_reducer.fit_transform(X)
        print('Dimensionality completed... reduced data shape:', reduced_X.shape)
        return reduced_X
    
    def run_sampling_strategy(self, X, y, model):
        self.Scumulative_X = np.empty((0, X.shape[1]))
        self.Scumulative_y = np.empty((0,))
        self.X_size = len(X)
        current_batch_X = np.empty((0, X.shape[1]))
        current_batch_y = np.empty((0,))

        if self.dim_reducer is not None:
            reduced_X = self.dim_reducer.fit_transform(X)
        else:
            reduced_X = X
        
        Q = self.strategy.get_distribution(reduced_X)
        
        # Separate set for Wasserstein distance computation
        distance_calc_set_X = np.empty((0, reduced_X.shape[1]))

        while not self.stopping_criterion():
            indices = self.strategy.sample(reduced_X, self.sampled_indices, self.subset_size)
            if len(indices) == 0:
                break
            self.sampled_indices.update(indices)
            
            St_X = X[indices]
            St_y = y[indices]
            St_reduced_X = reduced_X[indices]

            current_batch_X = np.concatenate([current_batch_X, St_X])
            current_batch_y = np.concatenate([current_batch_y, St_y])

            # Update Wasserstein distance set
            distance_calc_set_X = np.concatenate([distance_calc_set_X, St_reduced_X])

            # print(f"current_batch_X shape: {current_batch_X.shape}") 
            # print(f"dist_X {distance_calc_set_X.shape}")
            P = self.strategy.get_distribution(distance_calc_set_X)
            distance = self.strategy.compute_distance(P, Q)
            print('distance', distance)

            if self.update_criterion(distance):
                model.fit(current_batch_X, current_batch_y)
                self.Scumulative_X = np.concatenate([self.Scumulative_X, current_batch_X])
                self.Scumulative_y = np.concatenate([self.Scumulative_y, current_batch_y])
                current_batch_X = np.empty((0, X.shape[1]))
                current_batch_y = np.empty((0,))
                # Reset Wasserstein distance set
                distance_calc_set_X = np.empty((0, reduced_X.shape[1]))
            else:
                if len(self.Scumulative_X) > 0:
                    model.fit(self.Scumulative_X, self.Scumulative_y)

            self.last_distance = distance
            self.tau *= self.decay_factor
            self.t += 1
            
        return model
    
    def update_criterion(self, distance):
        if self.last_distance is None or distance > self.last_distance:
            return False
        diff_ratio = (self.last_distance - distance) / self.last_distance
        return diff_ratio <= self.tau # distributions within a pct threshold tau

    def stopping_criterion(self):
        return self.t >= self.max_iterations or len(self.Scumulative_X) >= self.X_size
