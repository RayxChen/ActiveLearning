import ot
import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import entropy


class SamplingStrategy(ABC):
    
    def __init__(self, dist_estimator=None):
        """Initialize with an optional distribution estimator."""
        self.dist_estimator = dist_estimator

    def sample(self, X, sampled_indices, subset_size):
        """Default sampling method."""
        remaining_indices = list(set(range(len(X))) - sampled_indices)
        if len(remaining_indices) < subset_size:
            indices = np.random.choice(remaining_indices, size=len(remaining_indices), replace=False)
        else:
            indices = np.random.choice(remaining_indices, size=subset_size, replace=False)
        return indices
    
    def get_distribution(self, data):
        """Default method to get distribution using the dist_estimator."""
        if self.dist_estimator is None:
            raise ValueError("dist_estimator must be set to use get_distribution.")
        return self.dist_estimator.get_empirical_distribution(data)
    

    @abstractmethod
    def compute_distance(self, P, Q):
        pass

    
class KLDivergenceStrategy(SamplingStrategy):
    def __init__(self, dist_estimator, agg_method='mean'):
        super().__init__(dist_estimator)
        self.Agg = DistanceAggregator(agg_method)

    def compute_distance(self, Ps, Qs):
        """
        Computes the Kullback-Leibler (KL) divergence between two 1D distributions.

        Parameters:
        P, Q: Tuples containing (histogram, bin_edges)
            - P[0] and Q[0]: Histograms of the two distributions.
            - P[1] and Q[1]: Bin edges of the histograms (not used in KL divergence).

        Returns:
        KL divergence between the distributions represented by P and Q.
        """

        distances = []
        for P, Q in zip(Ps, Qs):
            # Extract histograms from P and Q and ensure no zero values (clip to a small epsilon)
            P = np.clip(P[0], 1e-10, None)
            Q = np.clip(Q[0], 1e-10, None)
            distances.append(entropy(P, Q))
        distance = self.Agg.aggregate(distances)
        return distance


class WassersteinDistanceStrategy(SamplingStrategy):
    def __init__(self, dist_estimator, agg_method='mean'):
        super().__init__(dist_estimator)
        self.Agg = DistanceAggregator(agg_method)

    def compute_distance(self, Ps, Qs):
        """
        Computes the Earth Mover's Distance (EMD), also known as the Wasserstein distance,
        between two 1D distributions.

        Parameters:
        P, Q: Tuples containing (histogram, bin_edges)
            - P[0] and Q[0]: Histograms of the two distributions.
            - P[1] and Q[1]: Bin edges of the histograms.

        Returns:
        EMD between the distributions represented by P and Q.
        """

        # Extract histograms and bin edges from P and Q
        distances = []
        for P, Q in zip(Ps, Qs):
            P_hist, P_edges = P
            Q_hist, Q_edges = Q

            # Calculate the centers of the bins for both histograms
            P_centers = (P_edges[:-1] + P_edges[1:]) / 2
            Q_centers = (Q_edges[:-1] + Q_edges[1:]) / 2
            distances.append(ot.emd2_1d(P_centers, Q_centers, P_hist, Q_hist))

        distance = self.Agg.aggregate(distances)
             
        return distance


class DistanceAggregator:
    def __init__(self, aggregation_method='mean', p=2):
        self.aggregation_method = aggregation_method
        self.p = p
    
    def aggregate(self, distances):
        distances = np.array(distances)
        
        if self.aggregation_method == 'mean':
            return np.mean(distances)
        elif self.aggregation_method == 'geometric_mean':
            return np.exp(np.mean(np.log(distances + 1e-10)))  # Avoid log of zero
        elif self.aggregation_method == 'rms':
            return np.sqrt(np.mean(np.square(distances)))
        elif self.aggregation_method == 'max':
            return np.max(distances)
        elif self.aggregation_method == 'harmonic_mean':
            return len(distances) / np.sum(1.0 / (distances + 1e-10))  # Avoid division by zero
        elif self.aggregation_method == 'lp_norm':
            return np.sum(np.abs(distances) ** self.p) ** (1/self.p)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
