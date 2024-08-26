import numpy as np
import matplotlib.pyplot as plt


class DistributionEstimator:
    def __init__(self, method='freedman-diaconis'):
        self.method = method
        self.bins = None

    def calculate_bins(self, data):
        """
        Calculate the optimal number of bins for each dimension of the dataset using the specified method.
        
        Parameters:
        data (np.array): The dataset, can be 1D or multi-dimensional. 
        
        Returns:
        int or list of ints: The optimal number of bins for each dimension of the dataset.
        """
        def _calculate_bins_1d(data_1d, method):
            n = len(data_1d)
            
            if method == 'freedman-diaconis':
                iqr = np.percentile(data_1d, 75) - np.percentile(data_1d, 25)
                bins = int(np.ceil((data_1d.max() - data_1d.min()) / (2 * iqr * n ** (-1 / 3))))
                
            elif method == 'scott':
                bins = int(np.ceil((data_1d.max() - data_1d.min()) / (3.5 * np.std(data_1d) * n ** (-1 / 3))))
            
            else:
                raise ValueError("Method must be either 'freedman-diaconis' or 'scott'")
            
            return bins

        # Flatten the data if it has shape (B, 1)
        if data.ndim == 2 and data.shape[1] == 1:
            data = data.flatten()

        if data.ndim == 1:
            self.bins = _calculate_bins_1d(data, self.method)
        else:
            # Calculate bins for each dimension separately
            self.bins = []
            for i in range(data.shape[1]):
                self.bins.append(_calculate_bins_1d(data[:, i], self.method))
        
        return self.bins

    def get_empirical_distribution(self, data):
        """
        Estimate the probability distribution of the data using histograms for each dimension.
        
        Parameters:
        data (np.array): The dataset, can be 1D or multi-dimensional.
        
        Returns:
        list of tuples: Each tuple contains (hist, bin_edges) for each dimension.
        """
        # Flatten the data if it has shape (B, 1)
        if data.ndim == 2 and data.shape[1] == 1:
            data = data.flatten()

        bins = self.calculate_bins(data) if self.bins is None else self.bins
        
        if data.ndim == 1:
            hist, bin_edges = np.histogram(data, bins=bins, density=True)
            hist += np.finfo(float).eps
            return [(hist / np.sum(hist), bin_edges)]
        
        else:
            distributions = []
            for i in range(data.shape[1]):
                hist, bin_edges = np.histogram(data[:, i], bins=bins[i], density=True)
                hist += np.finfo(float).eps
                distributions.append((hist / np.sum(hist), bin_edges))
            return distributions

    def visualize_empirical_distribution(self, distributions):
        """
        Visualize the histograms for each dimension.
        
        Parameters:
        distributions (list of tuples): Each tuple contains (hist, bin_edges) for each dimension.
        """
        for i, (hist, bin_edges) in enumerate(distributions):
            plt.figure()
            plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'Histogram for Dimension {i+1}')
            plt.show()
