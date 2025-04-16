import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
import sys
import os

# Add the root directory to the path to access other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.agent import Agent
from rewards import Reward

class RobotSimple(Agent):
    """
    Simple robot agent that only optimizes for collision avoidance and goal reaching.
    Used for generating training data without active information gathering.
    """
    
    def __init__(self, dynamics, 
                 initial_state: Union[np.ndarray, torch.Tensor],
                 reward: Reward = None,
                 name: str = "robot",
                 color: str = "yellow",
                 planning_horizon: int = 15,
                 optimization_steps: int = 50,
                 safety_distance: float = 0.2,
                 goal_position: torch.Tensor = None):
        """
        Initialize simple robot agent.
        
        Args:
            dynamics: Dynamics model for state transitions
            initial_state: Initial state of the agent
            reward: Reward function for decision making
            name: Agent identifier
            color: Color for visualization
            planning_horizon: Number of steps to look ahead
            optimization_steps: Number of iterations for optimization
            safety_distance: Minimum safe distance to other agents
            goal_position: Target position to reach [x, y]
        """
        super().__init__(dynamics, initial_state, name, color)
        
        # Set reward function
        self.reward = reward
        
        # Planning parameters
        self.planning_horizon = planning_horizon
        self.optimization_steps = optimization_steps
        self.safety_distance = safety_distance
        
        # Set goal position
        if goal_position is None:
            goal_position = torch.tensor([0.0, -2.0])  # Default goal
        self.goal_position = goal_position
        
        # For storing predicted trajectory
        self.predicted_states = []
        self.predicted_actions = []
    
    def compute_safety_constraint(self, state: torch.Tensor, 
                               human_position: torch.Tensor) -> bool:
        """
        Check if a state satisfies safety constraints.
        
        Args:
            state: Robot state to check
            human_position: Position of human agent
            
        Returns:
            True if state is safe
        """
        # Extract positions from states
        robot_position = state[:2]
        
        # Compute distance to human
        distance = torch.norm(human_position - robot_position)
        
        # Check if minimum distance exceeds safety threshold
        return distance >= self.safety_distance
    
    def compute_control(self, t: int, state: torch.Tensor, 
                        environment_state: Dict = None) -> torch.Tensor:
        """
        Compute control action for the robot agent using MPC with safety constraints.
        
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
            
        # If no environment state is provided, use empty dictionary
        if environment_state is None:
            environment_state = {}
            
        # Initialize planning
        current_action = self.action.clone()
        best_action = current_action.clone()
        best_reward = float('-inf')
        
        # Store predicted trajectory for the best action
        best_predicted_states = []
        best_predicted_actions = []
        
        # Get human position if available
        human_position = None
        for agent_id, agent in environment_state.items():
            if "human" in agent_id:
                human_position = agent.position
                break
        
        # Optimization using random sampling with safety constraints
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
            
            # Check if trajectory is safe
            is_safe = True
            
            for h in range(self.planning_horizon):
                # Propagate state using dynamics
                simulated_state = self.dynamics(simulated_state, trial_action)
                
                # Store for trajectory
                simulated_states.append(simulated_state.clone())
                simulated_actions.append(trial_action.clone())  # Simplified: same action throughout
                
                # Check safety constraint if human is present
                if human_position is not None:
                    if not self.compute_safety_constraint(simulated_state, human_position):
                        is_safe = False
                        break
            
            # Skip unsafe trajectories
            if not is_safe:
                continue
            
            # Compute total reward
            total_reward = torch.tensor(0.0, dtype=torch.float32)
            
            # Add task reward
            for h in range(len(simulated_states) - 1):
                if self.reward is not None:
                    step_reward = self.reward(t + h, simulated_states[h], simulated_actions[h])
                else:
                    # Simple reward: distance to goal + control cost
                    dist_to_goal = torch.norm(simulated_states[h][:2] - self.goal_position)
                    control_cost = torch.sum(simulated_actions[h]**2)
                    step_reward = -dist_to_goal - 0.1 * control_cost
                
                total_reward += step_reward
            
            # Update best action if we found better reward
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = trial_action.clone()
                best_predicted_states = simulated_states
                best_predicted_actions = simulated_actions
        
        # Store predicted trajectory
        self.predicted_states = best_predicted_states
        self.predicted_actions = best_predicted_actions
        
        return best_action