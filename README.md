# Active Learning with Monte Carlo Integration

## Overview
This project maintains i.i.d. conditions between training and test data by controlling distribution distances during training. It combines Active Learning with Monte Carlo methods and Sinkhorn distances to:

1. Optimize training data order for i.i.d. conditions
2. Designed for high dimensional datasets
3. Integrate prediction-based selection seamlessly

## Key Features

- **Distribution Control**
  - Uses different distance metrics for distribution measurement
  - Orders training data to maintain i.i.d. conditions 
  - Adapts to distribution shifts
  - Theoretically grounded approach for Sinkhorn distance

- **Efficient Computation**
  - Monte Carlo rejection sampling for given distribution distance
  - Memory-efficient pairwise calculations
  - Scalable clustering with adjustable detail
  - GPU acceleration support

- **Smart Sample Selection**
  - Distribution-aware sampling strategy
  - Prediction-based selection criteria
  - Hybrid selection approach


## Examples

### Cumulative Monte Carlo 
![Cumulative Monte Carlo](/Examples/Plots/Sinkhorn_without_replacement.png)
- Critical distance vector for Monte Carlo rejection sampling

### Sinkhorn HDBSCAN Clustering
![Sinkhorn HDBSCAN Clustering](/Examples/Plots/hdbscan_clusters_umap.png)
- Hierarchical density-based clustering using sinkhorn distance
- High dimensional multimodal data visualization with UMAP
