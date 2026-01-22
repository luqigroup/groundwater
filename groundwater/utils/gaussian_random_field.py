"""
Gaussian Random Field with efficient score computation via FFT.

Adapted from: https://github.com/devitocodes/devito/blob/master/examples/cfd/09_Darcy_flow_equation.ipynb
"""

import math
import numpy as np
import numpy.fft as fft

__all__ = ["GaussianRandomField"]


class GaussianRandomField:
    """
    Class for generating realizations of a Gaussian random field.

    Parameters:
    -----------
    dim : int
        Dimensionality of the random field (e.g., 2 for 2D fields).
    size : int
        Size of the grid for the random field in each dimension.
    alpha : float, optional
        Power exponent that controls the smoothness of the field. Defaults to 2.
    tau : float, optional
        Scale parameter that influences the correlation length of the field.
        Defaults to 3.
    sigma : float, optional
        Standard deviation of the field. If None, it is automatically calculated
        based on 'alpha', 'tau', and 'dim'. Defaults to None.
    boundary : str, optional
        Type of boundary conditions for the field. Currently, only "periodic" is
        supported. Defaults to "periodic".
    """

    def __init__(
        self,
        dim: int,
        size: int,
        alpha: float = 2,
        tau: float = 3,
        sigma: float = None,
        boundary: str = "periodic",
    ) -> None:
        self.dim = dim
        self.size = size
        self.alpha = alpha
        self.tau = tau

        if dim != 2:
            raise ValueError("Only 2D fields are supported.")

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))
        self.sigma = sigma

        # Wavenumbers
        k_max = size // 2
        k = np.concatenate([np.arange(0, k_max), np.arange(-k_max, 0)])
        k_x, k_y = np.meshgrid(k, k, indexing="ij")

        # Power spectrum: λ(k) ∝ (4π²|k|² + τ²)^(-α)
        spectrum = (4 * math.pi**2 * (k_x**2 + k_y**2) + tau**2) ** (-alpha / 2)

        # sqrt(eigenvalues) for sampling
        self.sqrt_eig = size**2 * math.sqrt(2) * sigma * spectrum
        self.sqrt_eig[0, 0] = 0  # zero mean

        # Eigenvalues are sqrt_eig^2 (the covariance in Fourier space)
        self.eig = self.sqrt_eig**2

        # Inverse eigenvalues for score, with regularization to prevent blowup
        self.inv_eig = np.zeros_like(self.eig)
        # Only invert eigenvalues above a fraction of the maximum
        eig_threshold = 1e-6 * np.max(self.eig)
        mask = self.eig > eig_threshold
        self.inv_eig[mask] = 1.0 / self.eig[mask]

    def sample(self, N: int, normalize: bool = False) -> np.ndarray:
        """
        Generate N samples from the Gaussian random field.

        Parameters:
        -----------
        N : int
            Number of samples to generate.
        normalize : bool, optional
            If True, normalize the field to have 0.5 maximum. Defaults to False.

        Returns:
        --------
        samples : np.ndarray
            An array of shape (N, size, size) containing N realizations of the
            Gaussian random field.
        """
        # Generate random coefficients for the Fourier space with normal
        # distribution
        coeff = np.random.randn(N, self.size, self.size)

        # Multiply the random coefficients by the square root of the eigenvalues
        coeff = self.sqrt_eig * coeff

        # Compute the inverse Fourier transform and return the real part of the
        # field
        field = fft.ifftn(coeff, axes=(-2, -1)).real

        if normalize:
            # Normalize the field to have 0.5 maximum
            field /= np.max(field)
            field *= 0.5
        return field

    def score(self, M: np.ndarray) -> np.ndarray:
        """
        Compute score: ∇_M log p(M) = -K⁻¹ M

        Via FFT: -ifft(fft(M) / λ(k))
        Complexity: O(N² log N)
        """
        single = M.ndim == 2
        if single:
            M = M[np.newaxis]

        M_fft = fft.fftn(M, axes=(-2, -1))
        score = -fft.ifftn(self.inv_eig * M_fft, axes=(-2, -1)).real

        return score[0] if single else score

    def log_prob(self, M: np.ndarray) -> float:
        """Log p(M) = -0.5 Mᵀ K⁻¹ M + const (normalized)"""
        M_fft = fft.fftn(M)
        # Normalize by number of elements to get average log prob per pixel
        n_elements = M.size
        return (
            -0.5 * np.sum(np.abs(M_fft) ** 2 * self.inv_eig).real / n_elements
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    grf = GaussianRandomField(2, size=128, alpha=3, tau=3)
    samples = grf.sample(10)

    # Plot samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(samples[i], cmap="viridis", origin="lower")
        ax.set_title(f"Sample {i + 1}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f"GRF Samples (α={grf.alpha}, τ={grf.tau})", fontsize=14)
    plt.tight_layout()
    plt.savefig("grf_samples.png", dpi=150, bbox_inches="tight")

    # Score visualization
    score = grf.score(samples[0])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(samples[0], cmap="viridis", origin="lower")
    axes[0].set_title("Sample M")
    axes[0].axis("off")
    axes[1].imshow(score, cmap="RdBu_r", origin="lower")
    axes[1].set_title(r"Score $\nabla_M \log p(M)$")
    axes[1].axis("off")
    axes[2].imshow(np.abs(score), cmap="hot", origin="lower")
    axes[2].set_title("|Score|")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig("grf_score.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Samples: {samples.shape}, Score: {score.shape}")
