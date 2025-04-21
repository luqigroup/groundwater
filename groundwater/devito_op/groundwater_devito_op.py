# Adapted from https://github.com/devitocodes/devito/blob/master/examples/cfd/09_Darcy_flow_equation.ipynb

import numpy as np
import torch
from devito import (
    Eq,
    Function,
    Grid,
    Operator,
    TimeFunction,
    configuration,
    div,
    exp,
    grad,
    initialize_function,
    solve,
    Dimension,
    Abs,
    Inc,
    Max
)

# Set Devito logging level to ERROR to suppress excessive output
configuration["log-level"] = "ERROR"

# Number of timesteps for pseudo-time integration in the PDE solver
NUM_PSEUDO_TIMESTEPS = 50000


class GroundwaterEquation:
    """
    Class representing the groundwater flow equation based on Darcy's law.

    Attributes:
        size (int): Size of the grid.
        grid (Grid): Devito grid object representing the spatial domain.
        p (TimeFunction): Time-dependent pressure function on the grid.
        u (Function): Coefficient function representing permeability (log of
            permeability).
        f (Function): Source term function.
        lambda_adj (TimeFunction): Time-dependent adjoint variable function.
        f_adj (Function): Adjoint source term function.
        gradient (Function): Gradient function to store the computed gradient
            of the objective function.
        fwd_op (Operator): Forward PDE operator for time-stepping.
        adj_op (Operator): Adjoint PDE operator for time-stepping.
    """

    def __init__(self, size: int):
        """
        Initialize the GroundwaterEquation class.

        Args:
            size (int): The size of the grid.
        """
        self.size = size
        self.grid = Grid(
            shape=(size, size), extent=(1.0, 1.0), dtype=np.float32
        )

        # Time-dependent pressure variable p(x,t)
        self.p = TimeFunction(name="p", grid=self.grid, space_order=2)

        # Permeability (log-permeability) field u(x)
        self.u = Function(name="u", grid=self.grid, space_order=2)

        # Source term function f(x)
        self.f = Function(name="f", grid=self.grid, space_order=2)

        # Static p
        self.p0 = Function(name="p0", grid=self.grid, space_order=2)

        # Gradient function
        self.gradient = Function(name="gradient", grid=self.grid, space_order=2)

        # Place holder to check convergence
        self.r = Function(name="r", dimensions=(Dimension('rdim'),), shape=(1,))

        # Adjoint variable lambda(x,t) and adjoint source f_adj(x)
        self.lambda_adj = TimeFunction(
            name="lambda_adj",
            grid=self.grid,
            space_order=2,
        )
        self.f_adj = Function(name="f_adj", grid=self.grid, space_order=2)

        # Set up forward and adjoint PDE operators
        self.fwd_op = {
            False: self.setup_foward_pde(self.p, self.u, self.f),
            True: self.setup_foward_pde(self.p, self.u, self.f, lin=True)
        }
        self.adj_op = self.setup_adjoint_pde(
            self.lambda_adj,
            self.u,
            self.f_adj,
        )
        self.lin_f_op = self.setup_linf_op(self.p0, self.u, self.gradient, self.f)
        self.op_gradient = None

    def setup_foward_pde(
        self,
        p: TimeFunction,
        u: Function,
        f: Function,
        lin: bool = False,
    ) -> Operator:
        """
        Set up the forward PDE operator for the pressure equation.

        Args:
            p (TimeFunction): Time-dependent pressure function.
            u (Function): Log-permeability function.
            f (Function): Source term function.

        Returns:
            Operator: Devito operator for forward PDE time-stepping.
        """
        x, y = self.grid.dimensions
        mx, my = self.grid.shape[0] // 3, self.grid.shape[1] // 3
        t = self.grid.stepping_dim

        # The PDE: -∇ · (e^u(x) ∇p(x)) = f
        equation_p = Eq(-div(exp(u) * grad(p, shift=0.5), shift=-0.5), f)
        stencil = solve(equation_p, p)
        update = Eq(p.forward, stencil)

        # Boundary conditions
        if not lin:
            bc = [
                # p(x)|x2=0 = x1
                Eq(p[t + 1, x, 0], x * self.grid.spacing[0]),
                # p(x)|x2=1 = 1 - x1
                Eq(p[t + 1, x, y.symbolic_max], 1.0 - x * self.grid.spacing[0]),
                # ∂p(x)/∂x1|x1=0 = 0
                Eq(p[t + 1, -1, y], p[t + 1, 0, y]),
                # ∂p(x)/∂x1|x1=1 = 0
                Eq(p[t + 1, x.symbolic_max + 1, y], p[t + 1, x.symbolic_max, y]),
            ]
        else:
            bc = [
                # p(x)|x2=0 = 0
                Eq(p[t + 1, x, 0], 0),
                # p(x)|x2=1 = 0
                Eq(p[t + 1, x, y.symbolic_max], 0),
                # ∂p(x)/∂x1|x1=0 = 0
                Eq(p[t + 1, -1, y], p[t + 1, 0, y]),
                # ∂p(x)/∂x1|x1=1 = 0
                Eq(p[t + 1, x.symbolic_max + 1, y], p[t + 1, x.symbolic_max, y]),
            ]

        # Compute error in the middle
        residual = [Eq(self.r[0], 0),
                    Eq(self.r[0], Abs(equation_p.lhs.evaluate.subs({t: 1, x: mx, y: my}) -
                                      equation_p.rhs.subs({x: mx, y: my})))]

        op = Operator([update] + bc + residual)

        return op

    def setup_adjoint_pde(
        self,
        lambda_adj: TimeFunction,
        u: Function,
        f_adj: Function,
    ) -> Operator:
        """
        Set up the adjoint PDE operator for the adjoint equation.

        Args:
            lambda_adj (TimeFunction): Time-dependent adjoint variable.
            u (Function): Log-permeability function.
            f_adj (Function): Adjoint source term function.

        Returns:
            Operator: Devito operator for adjoint PDE time-stepping.
        """
        x, y = self.grid.dimensions
        t = self.grid.stepping_dim

        # Adjoint equation: -∇ · (e^u(x) ∇λ(x)) = f_adj
        adj_equation = Eq(
            -div(exp(u) * grad(lambda_adj, shift=-0.5), shift=0.5),
            f_adj,
        )
        stencil_adj = solve(adj_equation, lambda_adj)
        update_adj = Eq(lambda_adj.forward, stencil_adj)

        # Boundary conditions for the adjoint equation
        bc_adj = [
            # λ(x)|x2=0 = 0,
            Eq(lambda_adj[t + 1, x, 0], 0),
            # λ(x)|x2=1 = 0,
            Eq(lambda_adj[t + 1, x, y.symbolic_max], 0),
            # ∂λ(x)/∂x1|x1=0 = ∂λ(x)/∂x1|x1=1.
            Eq(lambda_adj[t + 1, 0, y], lambda_adj[t + 1, 1, y]),
            Eq(
                lambda_adj[t + 1, x.symbolic_max, y],
                lambda_adj[t + 1, x.symbolic_max - 1, y],
            ),
        ]

        return Operator([update_adj] + bc_adj)

    def setup_linf_op(self, p: Function, u: Function, du: Function, f: Function) -> Operator:
        """
        Set up the linearized forward PDE operator for the pressure equation.

        Args:
            p (TimeFunction): Time-dependent pressure function.
            u (Function): Log-permeability function.

        Returns:
            Operator: Devito operator for linearized forward PDE time-stepping.
        """
        # Linearized PDE f:  ∇ · (e^(u(x)) du(x) ∇p)
        return Operator(Eq(f,  div(exp(u) * du * grad(p, shift=0.5), shift=-0.5)))

    def eval_fwd_op(
        self,
        f: np.ndarray,
        u: np.ndarray,
        time_steps: int = NUM_PSEUDO_TIMESTEPS,
        lin: bool = False,
        return_array: bool = True,
    ) -> np.ndarray:
        """
        Evaluate the forward PDE operator for a given source and permeability
            field.

        Args:
            f (np.ndarray): Source term array.
            u (np.ndarray): Log-permeability array.
            time_steps (int, optional): Number of pseudo-timesteps for
                integration. Defaults to NUM_PSEUDO_TIMESTEPS.
            return_array (bool, optional): Whether to return the pressure data
                as a NumPy array. Defaults to True.

        Returns:
            np.ndarray: Pressure field after time-stepping.
        """
        initialize_function(self.f, f, 0)
        initialize_function(self.u, u, 0)
        self.p.data_with_halo.fill(0.0)
        self.fwd_op[lin](time=time_steps)

        print(f"Residual (lin: {lin}) at last iteration: {self.r.data[0]}")

        if return_array:
            return np.array(self.p.data[1])
        else:
            return self.p

    def eval_adj_op(
        self,
        u: np.ndarray,
        residual: np.ndarray,
        time_steps: int = NUM_PSEUDO_TIMESTEPS,
        return_array: bool = True,
    ) -> np.ndarray:
        """
        Evaluate the adjoint PDE operator for a given residual and permeability
            field.

        Args:
            u (np.ndarray): Log-permeability array.
            residual (np.ndarray): Residual array (difference between measured
                and simulated data).
            time_steps (int, optional): Number of pseudo-timesteps for
                integration. Defaults to NUM_PSEUDO_TIMESTEPS.
            return_array (bool, optional): Whether to return the adjoint
                variable data as a NumPy array. Defaults to True.

        Returns:
            np.ndarray: Adjoint variable field after time-stepping.
        """
        initialize_function(self.f_adj, residual, 0)
        initialize_function(self.u, u, 0)
        self.lambda_adj.data_with_halo.fill(0.0)
        self.adj_op(time=time_steps)

        if return_array:
            return np.array(self.lambda_adj.data[1])
        else:
            return self.lambda_adj

    def eval_f_lin(
        self,
        u: np.ndarray,
        du: np.ndarray,
        p_fwd: np.ndarray,
        return_array: bool = True) -> np.ndarray:
        """
        Evaluate the linearized forcing term for the forward PDE operator.
        Args:
            f (np.ndarray): Source term array.
            u (np.ndarray): Log-permeability array.
            du (np.ndarray): Perturbation in the log-permeability field.
            p_fwd (np.ndarray): Forward pressure field.
        Returns:
            np.ndarray: Linearized forcing term.
        """
        initialize_function(self.u, u, 0)
        initialize_function(self.gradient, du, 0)
        self.f.data_with_halo.fill(0.0)
        initialize_function(self.p0, p_fwd, 0)

        self.lin_f_op()

        if return_array:
            return np.array(self.f.data)
        else:
            return self.f

    def compute_gradient(
        self, u0: np.ndarray, residual: np.ndarray, p_fwd: TimeFunction
    ) -> np.ndarray:
        """
        Compute the gradient of the objective function with respect to the
            permeability field:  ∇_u J = -e^u ∇λ · ∇p

        Args:
            u0 (np.ndarray): Initial guess for the log-permeability
                field.
            residual (np.ndarray): Residual array (difference between measured
                and simulated data).
            p_fwd (TimeFunction): Forward pressure field.

        Returns:
            np.ndarray: Gradient of the objective function with respect to the
                permeability field.
        """
        self.gradient.data_with_halo.fill(0.0)
        initialize_function(self.u, u0, 0)

        # Evaluate adjoint variable
        lambda_adj = self.eval_adj_op(u0, residual, return_array=False)

        # -e^u ∇λ · ∇p term for gradient computation
        if self.op_gradient is None:
            t = self.grid.stepping_dim
            # Gradient of the objective function with respect to u
            grad_lambda = grad(lambda_adj, shift=0.5)._subs(t, 1)
            grad_p = grad(p_fwd, shift=0.5)._subs(t, 1)
            gradient_eq = Eq(
                self.gradient, -exp(self.u) * (grad_lambda.dot(grad_p))
            )
            self.op_gradient = Operator(gradient_eq)

        # Compute the gradient
        self.op_gradient()

        return np.array(self.gradient.data)

    def compute_linearization(self,
        f: np.ndarray,
        u: np.ndarray,
        du: np.ndarray,
        time_steps: int = NUM_PSEUDO_TIMESTEPS,
        return_array: bool = True,
    ) -> np.ndarray:
        # First pass: Compute the forward model with the original u
        p_fwd = self.eval_fwd_op(f, u, time_steps)

        # Compute linearized forcing
        f_lin = self.eval_f_lin(u, du, p_fwd)

        # Second pass: Compute the forward model with the linearized forcing
        p_fwd_lin = self.eval_fwd_op(f_lin, u, time_steps, lin=True,
                                     return_array=return_array)

        return p_fwd_lin


class GroundwaterLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, f, groundwater_eq):
        # Convert PyTorch tensors to NumPy arrays
        u_np = u.detach().cpu().numpy()
        f_np = f.detach().cpu().numpy()

        # Run the forward model
        p_fwd = groundwater_eq.eval_fwd_op(f_np, u_np, return_array=False)

        # Save variables needed for backward pass
        ctx.save_for_backward(u, f)
        ctx.groundwater_eq = groundwater_eq
        ctx.p_fwd = p_fwd

        # Convert the result back to a PyTorch tensor
        return torch.from_numpy(p_fwd.data[0]).to(u.device).clone()

    @staticmethod
    def backward(ctx, grad_output):
        u, f = ctx.saved_tensors
        groundwater_eq = ctx.groundwater_eq
        p_fwd = ctx.p_fwd

        # Convert grad_output to NumPy array
        grad_output_np = grad_output.cpu().numpy()

        # Compute the gradient using the saved variables
        gradient = groundwater_eq.compute_gradient(
            u.cpu().numpy(), grad_output_np, p_fwd
        )

        # Convert the gradient back to a PyTorch tensor
        grad_u = torch.from_numpy(gradient).to(u.device)

        # Return gradients for each input tensor (None for groundwater_eq as
        # it's not a tensor)
        return grad_u, None, None


class GroundwaterModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.groundwater_eq = GroundwaterEquation(size)

    def forward(self, u, f):
        return GroundwaterLayer.apply(u, f, self.groundwater_eq)
