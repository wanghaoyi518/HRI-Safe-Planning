import torch
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Callable
import sys
import os

# Add the root directory to the path to access other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.agent import Agent
from rewards import ParameterizedReward, HumanReward


class HumanAgent(Agent):
    """
    Human agent model with internal state.
    
    Models human behavior based on reward maximization with internal state
    parameters for attentiveness and driving style.
    """
    
    def __init__(self, dynamics, 
             initial_state: Union[np.ndarray, torch.Tensor],
             internal_state: Union[np.ndarray, torch.Tensor] = None,
             reward: ParameterizedReward = None,
             name: str = "human",
             color: str = "red",
             planning_horizon: int = 5,
             optimization_steps: int = 50):
        """
        Initialize human agent with dynamics and internal state.
        """
        super().__init__(dynamics, initial_state, name, color)
        
        # Set internal state (default to medium attentiveness, normal style)
        if internal_state is None:
            internal_state = torch.tensor([0.5, 0.5], dtype=torch.float32)
        elif isinstance(internal_state, np.ndarray):
            internal_state = torch.tensor(internal_state, dtype=torch.float32)
            
        self.internal_state = internal_state
        
        # Set reward function based on internal state if not provided
        if reward is None:
            from rewards import create_parameterized_human_reward
            reward = create_parameterized_human_reward(self.internal_state)
        
        self.reward = reward
        
        # Planning parameters
        self.planning_horizon = planning_horizon
        self.optimization_steps = optimization_steps
        
        # For storing predicted trajectory
        self.predicted_states = []
        self.predicted_actions = []
        
    @property
    def attentiveness(self) -> float:
        """Get attentiveness level."""
        return self.internal_state[0].item()
        
    @property
    def driving_style(self) -> float:
        """Get driving style (aggressiveness)."""
        return self.internal_state[1].item()
    
    def compute_control(self, t: int, state: torch.Tensor, 
                        environment_state: Dict = None) -> torch.Tensor:
        """
        Compute control action for the human agent.
        
        The human agent computes actions by optimizing their reward
        function over a finite horizon, considering the current state
        and predictions about the environment.
        
        Args:
            t: Current time step
            state: Current state
            environment_state: Dictionary with information about other agents
            
        Returns:
            Control action
        """
        # Update current state if it differs from the provided state
        if torch.any(self.state != state):
            self.state = state.clone()
            
        # If no environment state is provided, use a simplified model
        if environment_state is None:
            environment_state = {}
            
        # Initialize planning
        current_action = self.action.clone()
        best_action = current_action.clone()
        best_reward = float('-inf')
        
        # Store predicted trajectory for the best action
        best_predicted_states = []
        best_predicted_actions = []
        
        # Simple optimization using random sampling
        for _ in range(self.optimization_steps):
            # Sample a random action with noise around current best action
            action_noise = torch.randn_like(best_action) * 0.2
            trial_action = best_action + action_noise
            
            # Clamp to valid control range
            trial_action = torch.clamp(trial_action, -1.0, 1.0)
            
            # Simulate trajectory for planning horizon
            simulated_state = self.state.clone()
            simulated_states = [simulated_state.clone()]
            simulated_actions = [trial_action.clone()]
            
            total_reward = torch.tensor(0.0, dtype=torch.float32)
            
            for h in range(self.planning_horizon):
                # Compute reward for current state-action pair
                step_reward = self.reward(t + h, simulated_state, trial_action, self.internal_state)
                total_reward += step_reward
                
                # Propagate state using dynamics
                simulated_state = self.dynamics(simulated_state, trial_action)
                
                # Store for trajectory
                simulated_states.append(simulated_state.clone())
                
                # For simplicity, keep the same action (could be improved)
                simulated_actions.append(trial_action.clone())
            
            # Update best action if we found better reward
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = trial_action.clone()
                best_predicted_states = simulated_states
                best_predicted_actions = simulated_actions
        
        # Store predicted trajectory
        self.predicted_states = best_predicted_states
        self.predicted_actions = best_predicted_actions
        
        # Apply reaction time based on attentiveness
        # Less attentive drivers react more slowly
        reaction_delay = max(0.1, 1.0 - self.attentiveness)  # Minimum delay of 0.1
        
        # Mix between previous action and new best action based on reaction time
        final_action = (reaction_delay * self.action + 
                       (1.0 - reaction_delay) * best_action)
        
        return final_action
        
    def get_observation_likelihood(self, 
                                 state: torch.Tensor, 
                                 action: torch.Tensor,
                                 robot_action: torch.Tensor, 
                                 internal_state: torch.Tensor) -> torch.Tensor:
        """
        Compute likelihood of observing an action given a state and internal state.
        
        This is used for belief updates in Bayesian inference.
        
        Args:
            state: Current state
            action: Observed action
            robot_action: Robot's action
            internal_state: Trial internal state
            
        Returns:
            Likelihood of the observed action
        """
        # Set temporary internal state for likelihood computation
        original_internal_state = self.internal_state
        self.internal_state = internal_state
        
        # Compute "optimal" action for this internal state
        optimal_action = self.compute_control(0, state, {"robot_action": robot_action})
        
        # Compute difference between optimal and observed actions
        diff = torch.sum((optimal_action - action) ** 2)
        
        # Convert to likelihood using Gaussian model
        # Smaller differences -> higher likelihood
        likelihood = torch.exp(-10.0 * diff)
        
        # Restore original internal state
        self.internal_state = original_internal_state
        
        return likelihood


if __name__ == "__main__":
    # Test the HumanAgent class
    from dynamics import CarDynamics
    from rewards import *
    
    # Create dynamics model
    dynamics = CarDynamics(dt=0.1)
    
    # Create initial state: [x, y, theta, v]
    initial_state = torch.tensor([0.0, 0.0, 0.0, 0.5])
    
    # Create agents with different internal states
    human_agent = HumanAgent(
        dynamics, 
        initial_state.clone(),
        internal_state=torch.tensor([0.9, 0.5]),  # High attentiveness
        reward=create_parameterized_human_reward(internal_state=torch.tensor([0.5, 0.5])),
        name="attentive_human",
        color="green"
    )
    
    # Create environment state with a robot nearby
    env_state = {
        "robot_position": torch.tensor([0.1, 0.1]),
        "robot_action": torch.tensor([0.0, 0.0])
    }
    
    # Compute controls for both agents
    human_action = human_agent.compute_control(0, initial_state, env_state)
    
    print(f"Human action: {human_action}")
    
    # Simulate one step for the human agent
    human_agent.step(human_action)
    
    print(f"Human state after step: {human_agent.state}")
    
    # Test observation likelihood
    test_state = torch.tensor([0.0, 0.0, 0.0, 0.5])
    test_action = torch.tensor([0.1, 0.2])
    test_robot_action = torch.tensor([0.0, 0.0])
    
    # Compute likelihood for different internal states
    attentive_likelihood = human_agent.get_observation_likelihood(
        test_state, test_action, test_robot_action, 
        torch.tensor([0.9, 0.5])  # Attentive internal state
    )
    
    distracted_likelihood = human_agent.get_observation_likelihood(
        test_state, test_action, test_robot_action, 
        torch.tensor([0.2, 0.5])  # Distracted internal state
    )
    
    print(f"Likelihood for attentive internal state: {attentive_likelihood.item()}")
    print(f"Likelihood for distracted internal state: {distracted_likelihood.item()}")
