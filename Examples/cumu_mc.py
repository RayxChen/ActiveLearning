import torch
from MonteCarlo.estimator import MonteCarloEstimator
from torch.distributions import MultivariateNormal


def create_gaussian_samples(mean: torch.Tensor, cov: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Create samples from a multivariate Gaussian distribution."""
    dist = MultivariateNormal(mean, cov)
    return dist.sample((n_samples,))

def create_multimodal_gaussian_samples(n_modes: int, dim: int, n_samples: int) -> torch.Tensor:
    """Create samples from a mixture of multivariate Gaussian distributions where each dimension
    has independent means and variances."""
    means = torch.randn(n_modes, dim) * 2
    variances = torch.rand(n_modes, dim) + 0.5
    
    samples_per_mode = [n_samples // n_modes] * (n_modes - 1)
    samples_per_mode.append(n_samples - sum(samples_per_mode))
    
    all_samples = []
    for mode in range(n_modes):
        cov = torch.diag(variances[mode])
        dist = MultivariateNormal(means[mode], cov)
        mode_samples = dist.sample((samples_per_mode[mode],))
        all_samples.append(mode_samples)
    
    return torch.cat(all_samples, dim=0)

if __name__ == "__main__":

    # Test parameters
    N_SAMPLES = 2000
    DIM = 10
    SUBSET_SIZE = 50
    DEVICE = [0] if torch.cuda.is_available() else []
    
    # Generate samples
    samples = create_multimodal_gaussian_samples(
        n_modes=3,
        dim=DIM,
        n_samples=N_SAMPLES
    )

    # Initialize estimator with correct parameter structure
    estimator = MonteCarloEstimator(
        X=samples,
        subset_size=SUBSET_SIZE,
        device=DEVICE,
        metric_kwargs={  # Nest Sinkhorn parameters under metric_kwargs
            'p': 1,
            'blur': 0.05,
            'debias': True
        }
    )

    # Calculate statistics with and without replacement
    for replacement in [True, False]:
    # replacement = False
        stats, valid_indices = estimator.calculate_statistics(
            n_pass=30,
            k_sigma=2.0,
            replacement=replacement,
            plot_path=f'Plots/Sinkhorn_{"with" if replacement else "without"}_replacement.png'
        )
        
        # Print results
        estimator.print_results(stats, replacement)
