import torch
import umap
import numpy as np
from MonteCarlo.HDBSCAN import SinkhornHDBSCAN
from MonteCarlo.visualization import ClusterVisualizer


def generate_multimodal_gaussian(n_samples=10000, n_features=5, n_modes=3):
    """Generate multimodal Gaussian data"""
    # Generate centers for each mode
    centers = torch.tensor([
        [-1.0] * n_features,  # First mode center
        [0.0] * n_features,   # Second mode center
        [1.0] * n_features    # Third mode center
    ])
    
    # Initialize data tensor
    data = torch.empty((n_samples, n_features))
    samples_per_mode = n_samples // n_modes
    
    # Generate samples for each mode
    for i in range(n_modes):
        start_idx = i * samples_per_mode
        end_idx = start_idx + samples_per_mode
        # Add noise with increasing variance for each mode
        noise = torch.randn(samples_per_mode, n_features) * (0.5 + i * 0.2)
        data[start_idx:end_idx] = centers[i] + noise
    
    return data

def test_multiple_distributions():
    """Test clustering on multimodal distribution"""
    print("\nTesting multimodal distribution clustering...")
    
    # Generate multimodal data
    data = generate_multimodal_gaussian(
        n_samples=10000,
        n_features=20,
        n_modes=3
    )
    
    print(f"Data shape: {data.shape}")
    
    # Initialize and run clustering
    clusterer = SinkhornHDBSCAN(
        min_cluster_size=2,
        min_samples=2,
        cluster_selection_epsilon=0.0,
        p=1,
        blur=0.05,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Use subset size of 100 to get 100x100 distance matrix
    subset_size = 100  # This will create 100 subsets (10000/100 = 100)
    results = clusterer.fit_predict(data, subset_size=subset_size)
    
    # Print results with clearer formatting
    print(f"\nClustering Results:")
    print(f"Number of clusters found: {results.n_clusters}")
    print(f"Number of noise points: {results.noise_points}")
    print("\nCluster sizes (number of subsets in each cluster):")
    for cluster_id, size in results.cluster_sizes.items():
        print(f"Cluster {cluster_id}: {size} subsets")
    print(f"\nDistance matrix shape: {results.distance_matrix.shape}")

    # Prepare data for visualization using UMAP
    data_subsets = data.reshape(-1, subset_size, data.shape[1])
    data_means = data_subsets.mean(dim=1)

    # Initialize and fit UMAP with adjusted parameters
    n_samples = data_means.shape[0]
    n_neighbors = min(30, n_samples - 1)
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,  # Changed back to 2D for clearer visualization
        metric='euclidean',
        random_state=42,
        # verbose=True
    )
    embedding = reducer.fit_transform(data_means.numpy())

    # Handle any remaining inf/nan values
    embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

    # Create range datasets for visualization
    range_datasets = {}
    for cluster_id in range(results.n_clusters):
        cluster_mask = results.labels == cluster_id
        if cluster_mask.any():
            range_datasets[f'Cluster {cluster_id}'] = type('Dataset', (), {
                'data': torch.tensor(embedding[cluster_mask])
            })()
    
    # Add noise points if any
    noise_mask = results.labels == -1
    if noise_mask.any():
        range_datasets['Noise'] = type('Dataset', (), {
            'data': torch.tensor(embedding[noise_mask])
        })()

    # Visualize clusters
    visualizer = ClusterVisualizer()
    visualizer.plot_clusters(
        range_datasets,
        title="Sinkhorn HDBSCAN Clustering Results (UMAP projection)",
        save_path="./Plots/hdbscan_clusters_umap.html"
    )
    print("\nVisualization saved to ./Plots/hdbscan_clusters_umap.html")

def main():
    print("Starting HDBSCAN tests...")
    test_multiple_distributions()
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 