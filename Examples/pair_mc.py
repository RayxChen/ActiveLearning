import torch
from MonteCarlo.distance_calculator import DistanceCalculator

if __name__ == "__main__":
    # Configuration
    n_samples = 1000
    n_features = 5
    n_passes = 20
    subset_size = 100
    dtype = torch.float32
    device = [0] if torch.cuda.is_available() else []
    
    # Generate synthetic data: Gaussian mixture with 3 components
    centers = torch.tensor([[-2.0]*n_features, [0.0]*n_features, [2.0]*n_features], dtype=dtype)
    weights = torch.tensor([0.3, 0.3, 0.4])
    stds = torch.tensor([0.5, 1.0, 2.0])
    
    # Create the dataset
    components = torch.multinomial(weights, n_samples, replacement=True)
    noise = torch.randn(n_samples, n_features, dtype=dtype)
    X = centers[components] + noise * stds[components].unsqueeze(1)
    
    print(f"Generated dataset shape: {X.shape}")
    
    # Initialize calculator
    calculator = DistanceCalculator(
        X=X,
        n_passes=n_passes,
        dtype=dtype,
        devices=device,
        metric_kwargs={
            'p': 1,
            'blur': 0.05,
            'debias': True
        }
    )
    
    # Test pairwise distances with different batch sizes
    print("\nTesting pairwise distances:")
    
    # Without batching
    calculator.batch_size = None
    pairwise_distances_no_batch = calculator.calculate_distances(subset_size)
    print("\nWithout batching:")
    print(f"Pairwise distances shape: {pairwise_distances_no_batch.shape}")
    print(f"Mean distance: {pairwise_distances_no_batch.mean():.4f}")
    print(f"Std distance: {pairwise_distances_no_batch.std():.4f}")
    
    # With batching
    calculator.batch_size = 10
    pairwise_distances_batch = calculator.calculate_distances(subset_size)
    print("\nWith batching (batch_size=5):")
    print(f"Pairwise distances shape: {pairwise_distances_batch.shape}")
    print(f"Mean distance: {pairwise_distances_batch.mean():.4f}")
    print(f"Std distance: {pairwise_distances_batch.std():.4f}")
    
    # Verify results are the same
    if torch.allclose(pairwise_distances_no_batch, pairwise_distances_batch, rtol=1e-4):
        print("\nVerification: Batched and non-batched results match!")
    else:
        print("\nWarning: Batched and non-batched results differ!")
    
    # Print distance matrix structure
    n_subsets = n_samples // subset_size
    print(f"\nNumber of subsets: {n_subsets}")
    print(f"Number of pairwise distances: {len(pairwise_distances_no_batch)}")
    print("\nFirst few distances:")
    print(pairwise_distances_no_batch[:5])
    
    # Memory usage report
    print("\nMemory Usage:")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    torch.cuda.reset_peak_memory_stats()
