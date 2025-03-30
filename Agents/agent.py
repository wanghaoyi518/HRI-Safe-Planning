import torch
import numpy as np
from typing import List, Tuple, Optional, Union, Dict

class Agent:
    """
    Base class for all agents (humans and robots) in the system.
    """
    
    def __init__(self, dynamics, 
                 initial_state: Union[np.ndarray, torch.Tensor],
                 name: str = "unnamed_agent",
                 color: str = "blue"):
        """
        Initialize agent with dynamics model and initial state.
        
        Args:
            dynamics: Dynamics model for state transitions
            initial_state: Initial state of the agent
            name: Agent identifier
            color: Color for visualization
        """
        self.dynamics = dynamics
        
        # Convert initial state to tensor if needed
        if isinstance(initial_state, np.ndarray):
            initial_state = torch.tensor(initial_state, dtype=torch.float32)
            
        self.state = initial_state
        self.initial_state = initial_state.clone()
        self.name = name
        self.color = color
        
        # Initialize action
        self.action = torch.zeros(dynamics.nu, dtype=torch.float32)
        
        # Keep history of states and actions
        self.state_history = [self.state.clone()]
        self.action_history = []
        
    def reset(self):
        """Reset agent to initial state."""
        self.state = self.initial_state.clone()
        self.action = torch.zeros_like(self.action)
        self.state_history = [self.state.clone()]
        self.action_history = []
        
    def step(self, action: Optional[torch.Tensor] = None):
        """
        Update agent state based on action.
        
        Args:
            action: Control input, uses current action if None
            
        Returns:
            New state
        """
        # Use provided action or current action
        if action is not None:
            self.set_action(action)
            
        # Compute next state using dynamics
        next_state = self.dynamics(self.state, self.action)
        
        # Update state
        self.state = next_state
        
        # Record history
        self.state_history.append(self.state.clone())
        self.action_history.append(self.action.clone())
        
        return self.state
    
    def set_action(self, action: Union[np.ndarray, torch.Tensor]):
        """
        Set the agent's action.
        
        Args:
            action: Control input
        """
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32)
            
        self.action = action
        
    def compute_control(self, t: int, state: torch.Tensor) -> torch.Tensor:
        """
        Compute control action for the agent.
        To be implemented by subclasses.
        
        Args:
            t: Current time step
            state: Current state
            
        Returns:
            Control action
        """
        raise NotImplementedError("Subclasses must implement compute_control")
        
    @property
    def position(self) -> torch.Tensor:
        """Get agent position."""
        return self.state[:2]
        
    @property
    def velocity(self) -> float:
        """Get agent velocity."""
        return self.state[3].item()
        
    @property
    def heading(self) -> float:
        """Get agent heading."""
        return self.state[2].item()


if __name__ == "__main__":
    # This is for testing the Agent class
    from dynamics import CarDynamics
    
    # Create dynamics model
    dynamics = CarDynamics(dt=0.1)
    
    # Create initial state: [x, y, theta, v]
    initial_state = torch.tensor([0.0, 0.0, 0.0, 0.0])
    
    # This would raise NotImplementedError since compute_control is not implemented
    # agent = Agent(dynamics, initial_state)
    
    # We'll implement a simple agent for testing
    class TestAgent(Agent):
        def compute_control(self, t, state):
            # Simple constant control
            return torch.tensor([0.1, 0.2])
    
    # Create and test the agent
    test_agent = TestAgent(dynamics, initial_state)
    print(f"Initial state: {test_agent.state}")
    
    # Simulate a few steps
    for i in range(5):
        action = test_agent.compute_control(i, test_agent.state)
        test_agent.step(action)
        print(f"Step {i+1}: State = {test_agent.state}, Action = {action}")