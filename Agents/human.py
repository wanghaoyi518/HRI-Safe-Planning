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
        # CRITICAL FIX: Adjusted scaling factor to create more variation
        likelihood = torch.exp(-5.0 * diff)  # Changed from -10.0 to -5.0
        
        # Add small random noise to ensure different likelihoods
        likelihood = likelihood + torch.rand(1)[0] * 1e-4
        
        # DEBUGGING
        # print(f"Internal state: {internal_state}, Action diff: {diff.item():.5f}, Likelihood: {likelihood.item():.5f}")
        
        # Restore original internal state
        self.internal_state = original_internal_state
        
        return likelihood

class BinaryHumanAgent(HumanAgent):
    """
    Human agent model with binary internal state (distracted/attentive).
    Simplified version for clearer modeling and faster computation.
    """
    
    def __init__(self, dynamics, 
                 initial_state: Union[np.ndarray, torch.Tensor],
                 is_attentive: bool = True,
                 name: str = "human",
                 color: str = "red",
                 planning_horizon: int = 5,
                 optimization_steps: int = 30):
        """
        Initialize binary human agent.
        
        Args:
            dynamics: Dynamics model for state transitions
            initial_state: Initial state of the agent
            is_attentive: True for attentive, False for distracted
            name: Agent identifier
            color: Color for visualization
            planning_horizon: Number of steps to look ahead
            optimization_steps: Number of iterations for optimization
        """
        # Convert boolean to binary internal state
        internal_state = torch.tensor([1.0 if is_attentive else 0.0], dtype=torch.float32)
        
        # Create binary reward function
        from rewards import create_binary_human_reward
        reward = create_binary_human_reward(is_attentive)
        
        # Initialize parent class with binary internal state
        super().__init__(
            dynamics=dynamics,
            initial_state=initial_state,
            internal_state=internal_state,
            reward=reward,
            name=name,
            color=color,
            planning_horizon=planning_horizon,
            optimization_steps=optimization_steps
        )
        
        # Store binary state
        self.is_attentive = is_attentive
        
    @property
    def attentiveness(self) -> float:
        """Get attentiveness level (0 or 1)."""
        return 1.0 if self.is_attentive else 0.0
        
    @property
    def driving_style(self) -> float:
        """For binary model, driving style is tied to attentiveness."""
        # Attentive drivers are moderate, distracted drivers are conservative
        return 0.5 if self.is_attentive else 0.3
    
    def set_internal_state(self, is_attentive: bool):
        """
        Set the binary internal state.
        
        Args:
            is_attentive: True for attentive, False for distracted
        """
        self.is_attentive = is_attentive
        self.internal_state = torch.tensor([1.0 if is_attentive else 0.0], dtype=torch.float32)
        
        # Update reward function
        from rewards import create_binary_human_reward
        self.reward = create_binary_human_reward(is_attentive)
    
    def compute_control(self, t: int, state: torch.Tensor, 
                        environment_state: Dict = None) -> torch.Tensor:
        """
        Compute control action for binary human agent.
        Simplified version with clearer behavioral differences.
        
        Args:
            t: Current time step
            state: Current state
            environment_state: Dictionary with information about other agents
            
        Returns:
            Control action
        """
        # Update current state if needed
        if torch.any(self.state != state):
            self.state = state.clone()
            
        if environment_state is None:
            environment_state = {}
            
        # For distracted drivers, add reaction delay
        if not self.is_attentive:
            # 30% chance of no reaction (maintaining previous action)
            torch.manual_seed(int(t * 10) % 1000)
            if torch.rand(1).item() < 0.3:
                return self.action.clone()
        
        # Use parent class optimization with reduced steps for efficiency
        optimization_steps = self.optimization_steps if self.is_attentive else self.optimization_steps // 2
        
        # Store original optimization steps
        original_steps = self.optimization_steps
        self.optimization_steps = optimization_steps
        
        # Call parent compute_control
        action = super().compute_control(t, state, environment_state)
        
        # Restore original optimization steps
        self.optimization_steps = original_steps
        
        # For distracted drivers, add noise to actions
        if not self.is_attentive:
            torch.manual_seed(int(t * 7 + state[0].item() * 100) % 1000)
            noise = torch.randn_like(action) * 0.1
            action = action + noise
            action = torch.clamp(action, -1.0, 1.0)
        
        return action
    
    def get_observation_likelihood(self, 
                                    state: torch.Tensor, 
                                    action: torch.Tensor,
                                    robot_action: torch.Tensor, 
                                    binary_state: Union[int, float, torch.Tensor]) -> torch.Tensor:
        """
        Compute likelihood of observing an action given a binary internal state.
        
        Args:
            state: Current state
            action: Observed action
            robot_action: Robot's action
            binary_state: Trial internal state (0 or 1)
            
        Returns:
            Likelihood of the observed action
        """
        # Convert binary state to boolean
        if isinstance(binary_state, torch.Tensor):
            is_attentive_trial = binary_state.item() > 0.5
        else:
            is_attentive_trial = binary_state > 0.5
        
        # Temporarily set internal state for likelihood computation
        original_is_attentive = self.is_attentive
        original_internal_state = self.internal_state.clone()
        original_reward = self.reward
        
        # Set trial state
        self.set_internal_state(is_attentive_trial)
        
        # Compute "optimal" action for this internal state
        optimal_action = self.compute_control(0, state, {"robot_action": robot_action})
        
        # Compute difference between optimal and observed actions
        action_diff = torch.sum((optimal_action - action) ** 2)
        
        # Different noise models for attentive vs distracted
        if is_attentive_trial:
            # Attentive drivers have lower action variance
            variance = 0.05
        else:
            # Distracted drivers have higher action variance
            variance = 0.2
        
        # Compute likelihood using Gaussian model
        likelihood = torch.exp(-action_diff / (2 * variance))
        
        # Restore original state
        self.is_attentive = original_is_attentive
        self.internal_state = original_internal_state
        self.reward = original_reward
        
        return likelihood
    
    def get_binary_observation_likelihood(self,
                                         state: torch.Tensor,
                                         action: torch.Tensor,
                                         robot_action: torch.Tensor) -> Tuple[float, float]:
        """
        Get likelihoods for both binary states.
        
        Args:
            state: Current state
            action: Observed action
            robot_action: Robot's action
            
        Returns:
            Tuple of (likelihood_distracted, likelihood_attentive)
        """
        likelihood_distracted = self.get_observation_likelihood(state, action, robot_action, 0.0)
        likelihood_attentive = self.get_observation_likelihood(state, action, robot_action, 1.0)
        
        return (likelihood_distracted.item(), likelihood_attentive.item())

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

    # Test BinaryHumanAgent
    print("\n\n=== Testing BinaryHumanAgent ===")
    
    # Create dynamics model
    dynamics = CarDynamics(dt=0.1)
    
    # Create initial state
    initial_state = torch.tensor([0.0, 0.0, 0.0, 0.5])
    
    # Create binary agents
    attentive_human = BinaryHumanAgent(
        dynamics, 
        initial_state.clone(),
        is_attentive=True,
        name="attentive_human",
        color="green"
    )
    
    distracted_human = BinaryHumanAgent(
        dynamics, 
        initial_state.clone(),
        is_attentive=False,
        name="distracted_human",
        color="red"
    )
    
    # Create environment state
    env_state = {
        "robot_position": torch.tensor([0.1, 0.1]),
        "robot_action": torch.tensor([0.0, 0.0])
    }
    
    # Test control computation
    print("Testing control computation:")
    for t in range(5):
        attentive_action = attentive_human.compute_control(t, initial_state, env_state)
        distracted_action = distracted_human.compute_control(t, initial_state, env_state)
        
        print(f"Time {t}:")
        print(f"  Attentive: {attentive_action}")
        print(f"  Distracted: {distracted_action}")
    
    # Test observation likelihoods
    print("\nTesting observation likelihoods:")
    test_action = torch.tensor([0.1, 0.2])
    test_robot_action = torch.tensor([0.0, 0.0])
    
    # Get likelihoods from attentive human's perspective
    lik_dist, lik_att = attentive_human.get_binary_observation_likelihood(
        initial_state, test_action, test_robot_action
    )
    
    print(f"For action {test_action}:")
    print(f"  P(action | distracted) = {lik_dist:.4f}")
    print(f"  P(action | attentive) = {lik_att:.4f}")
    
    # Test with different action (large steering)
    test_action_2 = torch.tensor([0.8, 0.1])
    lik_dist_2, lik_att_2 = attentive_human.get_binary_observation_likelihood(
        initial_state, test_action_2, test_robot_action
    )
    
    print(f"\nFor action {test_action_2} (large steering):")
    print(f"  P(action | distracted) = {lik_dist_2:.4f}")
    print(f"  P(action | attentive) = {lik_att_2:.4f}")
    
    # Test state switching
    print("\nTesting state switching:")
    print(f"Initial state - Attentive: {attentive_human.is_attentive}")
    attentive_human.set_internal_state(False)
    print(f"After switch - Attentive: {attentive_human.is_attentive}")
    
    # Simulate one step
    print("\nSimulating one step:")
    attentive_human.set_internal_state(True)  # Reset to attentive
    
    action_att = attentive_human.compute_control(0, initial_state, env_state)
    attentive_human.step(action_att)
    
    action_dist = distracted_human.compute_control(0, initial_state, env_state)
    distracted_human.step(action_dist)
    
    print(f"Attentive state after step: {attentive_human.state}")
    print(f"Distracted state after step: {distracted_human.state}")
