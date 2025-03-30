import torch
import torch.nn as nn
from typing import Callable, Optional, Union, Tuple
import numpy as np

class Dynamics:
    """Base class for dynamical systems."""
    
    def __init__(self, nx: int, nu: int, f: Callable, dt: Optional[float] = None):
        """
        Initialize dynamics with state and control dimensions.
        
        Args:
            nx: Number of state dimensions
            nu: Number of control dimensions
            f: Function that computes state derivatives
            dt: Time step for integration (if None, f should return next state directly)
        """
        self.nx = nx  # State dimension
        self.nu = nu  # Control dimension
        self.dt = dt
        
        # If dt is provided, wrap f in Euler integration
        if dt is None:
            self.f = f
        else:
            def integrated_f(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
                """Integrate dynamics using Euler integration."""
                dx = f(x, u)
                return x + dt * dx
            self.f = integrated_f
            
    def __call__(self, x: Union[torch.Tensor, np.ndarray], 
                u: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Compute next state given current state and control.
        
        Args:
            x: Current state [batch_size, nx] or [nx]
            u: Control input [batch_size, nu] or [nu]
            
        Returns:
            Next state as torch.Tensor
        """
        # Convert inputs to torch tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(u, np.ndarray):
            u = torch.tensor(u, dtype=torch.float32)
            
        return self.f(x, u)


class CarDynamics(Dynamics):
    """
    Car dynamics model using simple point-mass bicycle model.
    
    State: [x, y, theta, v]
    Control: [steering, acceleration]
    """
    
    def __init__(self, dt: float = 0.1, friction: float = 1.0):
        """
        Initialize car dynamics.
        
        Args:
            dt: Time step for integration
            friction: Friction coefficient for velocity damping
        """
        def f(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            """
            Compute state derivatives for car dynamics.
            
            Args:
                x: State vector [x, y, theta, v]
                u: Control vector [steering, acceleration]
                
            Returns:
                State derivatives [dx, dy, dtheta, dv]
            """
            # Extract states and controls
            theta = x[..., 2]  # heading angle
            v = x[..., 3]      # velocity
            
            # Compute derivatives
            dx = torch.stack([
                v * torch.cos(theta),                # dx/dt
                v * torch.sin(theta),                # dy/dt
                v * u[..., 0],                       # dtheta/dt = v * steering
                u[..., 1] - friction * v             # dv/dt = acceleration - friction*v
            ], dim=-1)
            
            return dx
            
        super().__init__(nx=4, nu=2, f=f, dt=dt)
    
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Return control bounds for car dynamics.
        
        Returns:
            Tuple of (steering_bounds, acceleration_bounds)
        """
        return ((-1.0, 1.0), (-1.0, 2.0))


if __name__ == '__main__':
    # Test the dynamics
    dyn = CarDynamics(dt=0.1)
    
    # Test with single state/control
    x = torch.tensor([0.0, 0.0, 0.0, 1.0])  # [x, y, theta, v]
    u = torch.tensor([0.1, 0.5])            # [steering, acceleration]
    
    next_x = dyn(x, u)
    print(f"Single state test:")
    print(f"Current state: {x}")
    print(f"Control input: {u}")
    print(f"Next state: {next_x}\n")
    
    # Test with batch of states/controls
    batch_x = torch.stack([x, x + 1.0])
    batch_u = torch.stack([u, -u])
    
    batch_next_x = dyn(batch_x, batch_u)
    print(f"Batch test:")
    print(f"Current states:\n{batch_x}")
    print(f"Control inputs:\n{batch_u}")
    print(f"Next states:\n{batch_next_x}")
    
    # Test control bounds
    steering_bounds, accel_bounds = dyn.bounds()
    print(f"\nControl bounds:")
    print(f"Steering: {steering_bounds}")
    print(f"Acceleration: {accel_bounds}")