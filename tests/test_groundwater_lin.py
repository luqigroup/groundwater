import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from groundwater.devito_op import GroundwaterEquation
from groundwater.utils import GaussianRandomField

def gradient_test(
    groundwater_eq: GroundwaterEquation,
    u0: np.ndarray,
    f: np.ndarray,
    dx: np.ndarray,
    epsilon: float = 1e-2,
    maxiter: int = 10,
) -> tuple[list[float], list[float]]:
    """
    Perform a gradient test using a Taylor expansion to verify the accuracy of
    the computed gradient. This is done by perturbing the input u0 in the
    direction of dx and checking if the errors decrease as expected.

    Parameters:
        groundwater_eq (GroundwaterEquation): Instance of the
            GroundwaterEquation class.
        u0 (np.ndarray): Initial guess for the input field u(x).
        f (np.ndarray): Forcing term f(x). d_obs (np.ndarray): Observed data
            from the forward simulation.
        dx (np.ndarray): Perturbation direction for testing gradient accuracy.
        epsilon (float, optional): Initial step size for the perturbation.
            Default is 1e-2.
        maxiter (int, optional): Maximum number of iterations for halving the
            step size. Default is 10.

    Returns:
        tuple[list[float], list[float]]: Lists of first-order and second-order
            errors.
    """
    errors_first_order = []
    errors_second_order = []

    rate_first_order = []
    rate_second_order = []

    # Compute forward operator for the initial guess u0
    p_fwd = groundwater_eq.eval_fwd_op(f, u0)

    # Linearized forward operator J(u0) * du
    Jdu = groundwater_eq.compute_linearization(f, u0, dx)

    h = epsilon
    for j in range(maxiter):
        # Perturb u0 in the direction of dx by h
        du = u0 + h * dx
        # Compute the perturbed forward operator
        p_perturbed = groundwater_eq.eval_fwd_op(f, du)

        # Compute the difference between the perturbed and original forward operator
        dp = p_perturbed - p_fwd

        # First order error F(m0 + hdm) - F(m0)
        err1 = np.linalg.norm(dp.reshape(-1), 1)

        # Second order error F(m0 + hdm) - F(m0) - h * <g, dm>
        err2 = np.linalg.norm((dp - h * Jdu).reshape(-1), 1)

        # Print step size and errors for each iteration
        print(
            f"Step {j + 1}: Step size = {h:.5e}, First-order error "
            f"= {err1:.5e}, hdx = "
            f"{np.linalg.norm((h*Jdu).reshape(-1), 1):.5e}"
        )

        # Append errors for analysis
        errors_first_order.append(err1)
        errors_second_order.append(err2)
        rate_first_order.append(
            errors_first_order[j] / errors_first_order[max(0, j - 1)]
        )
        rate_second_order.append(
            errors_second_order[j] / errors_second_order[max(0, j - 1)]
        )

        print(
            f"Step {j + 1}: Step size = {h:.5e}, First-order error = {err1:.5e}"
            f", Second-order error = {err2:.5e} First-order rate = "
            f"{rate_first_order[j]:.5e}, Second-order rate = "
            f"{rate_second_order[j]:.5e}"
        )

        # Halve the step size for the next iteration
        h *= 0.5

    return errors_first_order, errors_second_order


# Example usage:
if __name__ == "__main__":
    size = 128
    epsilon = 8

    # Generate the true input field using a Gaussian random field
    u_true = GaussianRandomField(2, size, alpha=2, tau=4).sample(2)[0]

    # Create a smoothed initial guess by applying a Gaussian filter to the true
    # field
    u0 = gaussian_filter(u_true, sigma=3)
    # Set the perturbation direction (difference between true field and smoothed
    # guess)
    du = u_true - u0

    # Forcing term (e.g., external influences) is initialized as zeros
    f = np.zeros((size, size))

    # Initialize the Groundwater equation problem
    groundwater_eq = GroundwaterEquation(size)

    # Simulate the forward operator with the true and background input field
    p_true = groundwater_eq.eval_fwd_op(f, u_true)
    p_smoothf = groundwater_eq.eval_fwd_op(f, u0, return_array=False)
    p_smooth = np.array(p_smoothf.data[1])
    print(f"dp = {np.linalg.norm((p_smooth-p_true).reshape(-1)):.5e}")
    # Simulate the forward/adjoint jacobian operator with the true input field
    # < p_true, J du> = < J^T p_true, du>
    gp = groundwater_eq.compute_gradient(u0, p_true, p_smoothf)
    dp = groundwater_eq.compute_linearization(f, u0, du)
    print(f"dp = {np.linalg.norm(dp.reshape(-1)):.5e}")
    term1 = np.dot(dp.reshape(-1), p_true.reshape(-1))
    term2 = np.dot(du.reshape(-1), gp.reshape(-1))
    print(f"Adjoint property: {term1:.5e}, {term2:.5e}, ratio = {term1/term2:.5e}")

    # Perform a gradient test
    errors_first_order, errors_second_order = gradient_test(
        groundwater_eq, u0, f, du, epsilon=epsilon, maxiter=5
    )

    # Estimate the expected error decay for first and second order
    h = [.95*errors_first_order[0] * 0.5**i for i in range(5)]
    h2 = [.95*errors_second_order[0] * 0.5 ** (2 * i) for i in range(5)]

    # Plot the errors (log-log scale) for visual comparison
    plt.semilogy(errors_first_order, '-*', label="First-order error", base=2)
    plt.semilogy(errors_second_order, '-*', label="Second-order error", base=2)
    plt.semilogy(h, '--o', label="h^1", base=2, markerfacecolor='none')
    plt.semilogy(h2, '--o', label="h^2", base=2, markerfacecolor='none')
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Gradient Test: Error vs Step Size")
    plt.grid(True)
    plt.show()
