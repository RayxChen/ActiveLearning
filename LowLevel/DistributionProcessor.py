import numpy as np
import matplotlib.pyplot as plt


class DistributionEstimator:
    def __init__(self, method='freedman-diaconis'):
        self.method = method
        self.bins = None
        self.bin_edges = None

    def calculate_bins(self, data):
        """
        Calculate the optimal number of bins and bin edges for each dimension of the dataset using the specified method.
        
        Parameters:
        data (np.array): The dataset, can be 1D or multi-dimensional. 
        
        Returns:
        list of np.array: The bin edges for each dimension of the dataset.
        """
        def _calculate_bins_1d(data_1d, method):
            n = len(data_1d)
            
            if method == 'freedman-diaconis':
                iqr = np.percentile(data_1d, 75) - np.percentile(data_1d, 25)
                bin_width = 2 * iqr * n ** (-1 / 3)
                
            elif method == 'scott':
                bin_width = 3.5 * np.std(data_1d) * n ** (-1 / 3)
                
            else:
                raise ValueError("Method must be either 'freedman-diaconis' or 'scott'")
            
            # Calculate bin edges based on bin width
            bin_edges = np.arange(data_1d.min(), data_1d.max() + bin_width, bin_width)
            return bin_edges

        # Flatten the data if it has shape (B, 1)
        if data.ndim == 2 and data.shape[1] == 1:
            data = data.flatten()

        if data.ndim == 1:
            self.bin_edges = _calculate_bins_1d(data, self.method)
        else:
            # Calculate bin edges for each dimension separately
            self.bin_edges = []
            for i in range(data.shape[1]):
                self.bin_edges.append(_calculate_bins_1d(data[:, i], self.method))
        
        return self.bin_edges

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

        if self.bin_edges is None:
            self.calculate_bins(data)
        
        if data.ndim == 1:
            hist, _ = np.histogram(data, bins=self.bin_edges, density=True)
            hist += np.finfo(float).eps
            return [(hist / np.sum(hist), self.bin_edges)]
        
        else:
            distributions = []
            for i in range(data.shape[1]):
                hist, _ = np.histogram(data[:, i], bins=self.bin_edges[i], density=True)
                hist += np.finfo(float).eps
                distributions.append((hist / np.sum(hist), self.bin_edges[i]))
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


