import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
import sys
import os

# Add the root directory to the path to access other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.agent import Agent
from Agents.belief_models import IntervalBelief, Interval
from rewards import RobotReward


class RobotAgent(Agent):
    """
    Robot agent that performs active information gathering and safe planning.
    """
    
    def __init__(self, dynamics, 
                 initial_state: Union[np.ndarray, torch.Tensor],
                 reward: RobotReward = None,
                 name: str = "robot",
                 color: str = "yellow",
                 planning_horizon: int = 5,
                 optimization_steps: int = 20,
                 info_gain_weight: float = 1.0,
                 safety_distance: float = 0.2):
        """
        Initialize robot agent.
        
        Args:
            dynamics: Dynamics model for state transitions
            initial_state: Initial state of the agent
            reward: Reward function for decision making
            name: Agent identifier
            color: Color for visualization
            planning_horizon: Number of steps to look ahead
            optimization_steps: Number of iterations for optimization
            info_gain_weight: Weight for information gain in reward
            safety_distance: Minimum safe distance to other agents
        """
        super().__init__(dynamics, initial_state, name, color)
        
        # Set reward function
        self.reward = reward
        
        # Planning parameters
        self.planning_horizon = planning_horizon
        self.optimization_steps = optimization_steps
        self.info_gain_weight = info_gain_weight
        self.safety_distance = safety_distance
        
        # Initialize belief over human internal state
        self.belief = IntervalBelief()
        
        # Store human models and predicted trajectories
        self.human_models = {}
        self.human_predicted_trajectories = {}
        
        # For storing reachable sets
        self.reachable_sets = []
        
        # For active information gathering
        self.entropy_history = []
        self.belief_convergence_data = []
        self.reachability_analyzer = None
        self.adaptive_planner = None
        
    def register_human(self, human_agent, model_id: str = None):
        """
        Register a human agent for interaction.
        
        Args:
            human_agent: Human agent to interact with
            model_id: Identifier for the human agent (default: agent name)
        """
        if model_id is None:
            model_id = human_agent.name
            
        self.human_models[model_id] = human_agent
        
    def update_belief(self, 
                     human_id: str,
                     observation: torch.Tensor,
                     num_samples: int = 10) -> None:
        """
        Update belief over human internal state based on observed actions.
        
        Args:
            human_id: Identifier for the human agent
            observation: Observed human action
            num_samples: Number of samples per interval
        """
        if human_id not in self.human_models:
            raise ValueError(f"Unknown human agent: {human_id}")
            
        human_agent = self.human_models[human_id]
        
        # Store initial entropy for information gain calculation
        initial_entropy = self.belief.entropy()
        
        # Define likelihood function
        def likelihood_fn(phi):
            return human_agent.get_observation_likelihood(
                human_agent.state,    # Current human state
                observation,          # Observed human action
                self.action,          # Robot action
                phi                   # Internal state sample
            )
            
        # Update belief
        self.belief.update(likelihood_fn, num_samples)
        
        # Calculate information gain
        final_entropy = self.belief.entropy()
        info_gain = initial_entropy - final_entropy
        
        # Store data for tracking convergence
        self.entropy_history.append(final_entropy.item())
        
        expected_state = self.belief.expected_value()
        self.belief_convergence_data.append({
            'expected': expected_state.clone(),
            'entropy': final_entropy.item(),
            'info_gain': info_gain.item()
        })
        
        # Refine intervals if needed based on threshold
        self.belief.refine(threshold=0.3)
        
    def compute_reachable_sets(self, 
                             human_id: str,
                             time_horizon: int = 5,
                             num_samples: int = 20) -> List[torch.Tensor]:
        """
        Compute reachable sets for human agent using sampling-based approach.
        
        Args:
            human_id: Identifier for the human agent
            time_horizon: Time horizon for prediction
            num_samples: Number of samples from belief distribution
            
        Returns:
            List of tensors representing reachable sets at each time step
        """
        if human_id not in self.human_models:
            raise ValueError(f"Unknown human agent: {human_id}")
            
        human_agent = self.human_models[human_id]
        
        # Sample internal states from belief distribution
        internal_state_samples = self.belief.sample(num_samples)
        
        # Initialize reachable sets
        reachable_sets = []
        
        # Store all simulated trajectories for each sampled internal state
        all_trajectories = []
        
        # For each sampled internal state
        for phi in internal_state_samples:
            # Set temporary internal state
            original_internal_state = human_agent.internal_state
            human_agent.internal_state = phi
            
            # Simulate trajectory
            simulated_states = [human_agent.state.clone()]
            simulated_state = human_agent.state.clone()
            
            for t in range(time_horizon):
                # Compute action for this internal state
                action = human_agent.compute_control(
                    t, 
                    simulated_state,
                    {"robot_state": self.state, "robot_action": self.action}
                )
                
                # Propagate state
                simulated_state = human_agent.dynamics(simulated_state, action)
                simulated_states.append(simulated_state.clone())
            
            # Reset human agent's internal state
            human_agent.internal_state = original_internal_state
            
            # Store trajectory
            all_trajectories.append(simulated_states)
        
        # Compute reachable sets at each time step
        for t in range(time_horizon + 1):
            # Collect states at time t from all trajectories
            states_at_t = [traj[t] for traj in all_trajectories]
            
            # Store as tensor
            reachable_sets.append(torch.stack(states_at_t))
        
        # Store for later use
        self.reachable_sets = reachable_sets
        
        return reachable_sets
    
    def compute_safety_constraint(self, 
                               state: torch.Tensor, 
                               reachable_set: torch.Tensor) -> bool:
        """
        Check if a state satisfies safety constraints with respect to reachable set.
        
        Args:
            state: Robot state to check
            reachable_set: Set of possible human states
            
        Returns:
            True if state is safe
        """
        # Extract positions from states
        robot_position = state[:2]
        
        # Extract human positions from reachable set
        human_positions = reachable_set[:, :2]
        
        # Compute distances to all points in reachable set
        distances = torch.norm(human_positions - robot_position, dim=1)
        
        # Check if minimum distance exceeds safety threshold
        return torch.min(distances) >= self.safety_distance
    
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
            
        # Extract human agent information
        human_id = next(iter(self.human_models.keys())) if self.human_models else None
        
        if human_id is None or human_id not in environment_state:
            # No human agent registered or no information available
            # Use simple control strategy
            return torch.zeros(self.dynamics.nu, dtype=torch.float32)
        
        # Compute reachable sets for human
        reachable_sets = self.compute_reachable_sets(human_id)
        
        # Initialize planning
        current_action = self.action.clone()
        best_action = current_action.clone()
        best_reward = float('-inf')
        
        # Initialize entropy before planning
        entropy_before = self.belief.entropy()
        
        # Store predicted trajectory for the best action
        best_predicted_states = []
        best_predicted_actions = []
        
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
            
            for h in range(min(self.planning_horizon, len(reachable_sets) - 1)):
                # Propagate state using dynamics
                simulated_state = self.dynamics(simulated_state, trial_action)
                
                # Store for trajectory
                simulated_states.append(simulated_state.clone())
                simulated_actions.append(trial_action.clone())  # Simplified: same action throughout
                
                # Check safety constraint
                if not self.compute_safety_constraint(simulated_state, reachable_sets[h + 1]):
                    is_safe = False
                    break
            
            # Skip unsafe trajectories
            if not is_safe:
                continue
            
            # Compute total reward including information gain
            total_reward = torch.tensor(0.0, dtype=torch.float32)
            
            # Simulate belief update for information gain calculation
            if human_id in environment_state:
                # Get human action from environment state
                human_action = environment_state[human_id].action
                
                # Create a copy of current belief
                temp_belief = IntervalBelief()
                temp_belief.intervals = self.belief.intervals.copy()
                temp_belief.probs = self.belief.probs.clone()
                
                # Update belief based on robot action and human reaction
                human_agent = self.human_models[human_id]
                
                def likelihood_fn(phi):
                    return human_agent.get_observation_likelihood(
                        human_agent.state,
                        human_action,
                        trial_action,  # Using trial action instead of current action
                        phi
                    )
                
                # Update temporary belief
                temp_belief.update(likelihood_fn)
                
                # Compute entropy after update
                entropy_after = temp_belief.entropy()
                
                # Information gain is reduction in entropy
                info_gain = entropy_before - entropy_after
                
                # Add to reward using info_gain_weight
                total_reward += self.info_gain_weight * info_gain
            
            # Add task reward
            for h in range(len(simulated_states) - 1):
                step_reward = self.reward(t + h, simulated_states[h], simulated_actions[h])
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
    
    def plan_with_adaptive_intervals(self, 
                                    t: int, 
                                    state: torch.Tensor,
                                    environment_state: Dict,
                                    total_samples: int = 100,
                                    refinement_threshold: float = 0.3) -> torch.Tensor:
        """
        Plan using adaptive interval-based reachability analysis (Algorithm 2).
        
        Args:
            t: Current time step
            state: Current state
            environment_state: Dictionary with information about other agents
            total_samples: Total number of samples to allocate
            refinement_threshold: Threshold for interval refinement
            
        Returns:
            Planned control action
        """
        # Extract human agent information
        human_id = next(iter(self.human_models.keys())) if self.human_models else None
        
        if human_id is None or human_id not in environment_state:
            return torch.zeros(self.dynamics.nu, dtype=torch.float32)
        
        human_agent = self.human_models[human_id]
        
        # Allocate samples based on belief
        sample_allocation = self.belief.sample_proportionally(total_samples)
        
        # Initialize empty reachable sets for each time step
        reachable_sets = [[] for _ in range(self.planning_horizon + 1)]
        
        # Perform reachability analysis for each interval
        for interval_idx, num_samples in sample_allocation.items():
            if num_samples <= 0:
                continue
                
            interval = self.belief.intervals[interval_idx]
            
            # Sample uniformly from this interval
            phi_samples = interval.uniform_sample(num_samples)
            
            # For each sampled phi, compute trajectory
            for phi in phi_samples:
                # Set temporary internal state
                original_internal_state = human_agent.internal_state
                human_agent.internal_state = phi
                
                # Simulate trajectory
                simulated_state = human_agent.state.clone()
                
                # Add initial state to reachable set
                reachable_sets[0].append(simulated_state.clone())
                
                # Propagate for planning horizon
                for h in range(self.planning_horizon):
                    # Compute action for this internal state
                    action = human_agent.compute_control(
                        t + h, 
                        simulated_state,
                        {"robot_state": state, "robot_action": self.action}
                    )
                    
                    # Propagate state
                    simulated_state = human_agent.dynamics(simulated_state, action)
                    
                    # Add to reachable set for this time step
                    reachable_sets[h + 1].append(simulated_state.clone())
                
                # Reset human agent's internal state
                human_agent.internal_state = original_internal_state
        
        # Convert reachable sets to tensors
        for h in range(len(reachable_sets)):
            if reachable_sets[h]:
                reachable_sets[h] = torch.stack(reachable_sets[h])
            else:
                # If no samples for this time step, use a default safe value
                reachable_sets[h] = torch.tensor([float('inf'), float('inf'), 0.0, 0.0]).unsqueeze(0)
        
        # Store reachable sets
        self.reachable_sets = reachable_sets
        
        # Plan safe trajectory using reachable sets (similar to compute_control but separated for clarity)
        current_action = self.action.clone()
        best_action = current_action.clone()
        best_reward = float('-inf')
        
        # Store predicted trajectory for the best action
        best_predicted_states = []
        best_predicted_actions = []
        
        # Optimization using random sampling with safety constraints
        for _ in range(self.optimization_steps):
            # Sample action with noise
            action_noise = torch.randn_like(best_action) * 0.2
            trial_action = best_action + action_noise
            
            # Clamp to valid control range
            trial_action = torch.clamp(trial_action, -1.0, 1.0)
            
            # Simulate trajectory
            simulated_state = self.state.clone()
            simulated_states = [simulated_state.clone()]
            simulated_actions = [trial_action.clone()]
            
            # Check if trajectory is safe
            is_safe = True
            
            for h in range(min(self.planning_horizon, len(reachable_sets) - 1)):
                # Propagate state
                simulated_state = self.dynamics(simulated_state, trial_action)
                
                # Store state and action
                simulated_states.append(simulated_state.clone())
                simulated_actions.append(trial_action.clone())
                
                # Check safety constraint
                if not self.compute_safety_constraint(simulated_state, reachable_sets[h + 1]):
                    is_safe = False
                    break
            
            # Skip unsafe trajectories
            if not is_safe:
                continue
            
            # Compute reward (includes information gain component via self.reward)
            total_reward = torch.tensor(0.0, dtype=torch.float32)
            
            for h in range(len(simulated_states) - 1):
                step_reward = self.reward(t + h, simulated_states[h], simulated_actions[h])
                total_reward += step_reward
            
            # Update best action if found better reward
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = trial_action.clone()
                best_predicted_states = simulated_states
                best_predicted_actions = simulated_actions
        
        # Store predicted trajectory
        self.predicted_states = best_predicted_states
        self.predicted_actions = best_predicted_actions
        
        return best_action
    
    def integrate_reachability_analyzer(self, reachability_analyzer):
        """
        Integrate reachability analyzer with robot planning.
        
        Args:
            reachability_analyzer: Instance of SamplingBasedReachabilityAnalysis
        """
        self.reachability_analyzer = reachability_analyzer
    
    def integrate_adaptive_planner(self, adaptive_planner):
        """
        Integrate adaptive interval planner with robot.
        
        Args:
            adaptive_planner: Instance of AdaptiveIntervalPlanner
        """
        self.adaptive_planner = adaptive_planner
    
    def get_max_probability_internal_state(self) -> torch.Tensor:
        """
        Get the most likely human internal state according to current belief.
        
        Returns:
            Most likely internal state
        """
        max_interval, _ = self.belief.max_probability_interval()
        return max_interval.center()
    
    def get_expected_internal_state(self) -> torch.Tensor:
        """
        Get the expected internal state according to current belief.
        
        Returns:
            Expected internal state
        """
        return self.belief.expected_value()
    
    def get_entropy_history(self) -> List[float]:
        """
        Get history of belief entropy.
        
        Returns:
            List of entropy values
        """
        return self.entropy_history
    
    def get_belief_convergence_data(self) -> List[Dict]:
        """
        Get data on belief convergence for analysis.
        
        Returns:
            List of dictionaries with convergence data
        """
        return self.belief_convergence_data
    
    def plot_belief(self, ax=None, show_center=True):
        """
        Plot the current belief distribution.
        
        Args:
            ax: Matplotlib axis (creates new figure if None)
            show_center: Whether to show expected value
        """
        return self.belief.plot(ax=ax, show_center=show_center)
    
    def plot_belief_history(self, true_state=None, figsize=(15, 10), max_plots=6):
        """
        Plot history of belief updates.
        
        Args:
            true_state: True human internal state (optional)
            figsize: Figure size
            max_plots: Maximum number of history states to plot
        """
        return self.belief.plot_history(true_state=true_state, figsize=figsize, max_plots=max_plots)
    
    def plot_reachable_sets(self, ax=None):
        """
        Plot the reachable sets for visualization.
        
        Args:
            ax: Matplotlib axis (creates new figure if None)
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        if not self.reachable_sets:
            ax.text(0.5, 0.5, "No reachable sets computed yet",
                   ha='center', va='center', transform=ax.transAxes)
            return ax
            
        # Plot robot position
        ax.plot(self.state[0].item(), self.state[1].item(), 'yo', markersize=10, label='Robot')
        
        # Plot reachable sets with different colors for each time step
        colors = ['lightblue', 'blue', 'darkblue', 'purple', 'red']
        
        for t, reachable_set in enumerate(self.reachable_sets):
            if t >= len(colors):
                break
                
            # Extract positions
            positions = reachable_set[:, :2].detach().numpy()
            
            # Plot scatter points
            ax.scatter(positions[:, 0], positions[:, 1], 
                      c=colors[t], alpha=0.5, 
                      label=f't+{t}' if t > 0 else 't')
            
            # Compute convex hull for visualization
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(positions)
                
                # Plot convex hull
                for simplex in hull.simplices:
                    ax.plot(positions[simplex, 0], positions[simplex, 1], 
                           c=colors[t], alpha=0.7)
            except:
                # Skip convex hull if scipy not available or points are collinear
                pass
                
        # Set labels and legend
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Reachable Sets for Human Agent')
        ax.legend()
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        return ax
    
    def plot_entropy_history(self, ax=None):
        """
        Plot history of belief entropy.
        
        Args:
            ax: Matplotlib axis (creates new figure if None)
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        if not self.entropy_history:
            ax.text(0.5, 0.5, "No entropy history data available",
                   ha='center', va='center', transform=ax.transAxes)
            return ax
            
        # Plot entropy over time
        ax.plot(self.entropy_history, 'b-', marker='o')
        
        # Set labels
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Belief Entropy')
        ax.set_title('Entropy Reduction Over Time')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return ax


if __name__ == "__main__":
    # Test the RobotAgent class
    from dynamics import CarDynamics
    from Agents.human import HumanAgent
    from rewards import create_robot_reward, create_attentive_reward
    
    # Create dynamics model
    dynamics = CarDynamics(dt=0.1)
    
    # Create initial states
    robot_init_state = torch.tensor([0.0, 0.0, 0.0, 0.0])
    human_init_state = torch.tensor([1.0, 0.0, np.pi, 0.5])
    
    # Create human agent
    human = HumanAgent(
        dynamics,
        human_init_state,
        internal_state=torch.tensor([0.8, 0.5]),  # High attentiveness
        reward=create_attentive_reward(),
        name="human"
    )
    
    # Create robot agent
    robot = RobotAgent(
        dynamics,
        robot_init_state,
        reward=create_robot_reward(info_gain_weight=2.0),
        name="robot",
        info_gain_weight=2.0
    )
    
    # Register human with robot
    robot.register_human(human)
    
    # Create environment state
    env_state = {
        "human": human
    }
    
    # Compute control for robot
    robot_action = robot.compute_control(0, robot_init_state, env_state)
    print(f"Robot action: {robot_action}")
    
    # Compute control for human
    human_action = human.compute_control(0, human_init_state, {"robot": robot})
    print(f"Human action: {human_action}")
    
    # Update robot belief based on human action
    robot.update_belief("human", human_action)
    
    # Get max probability internal state
    max_prob_state = robot.get_max_probability_internal_state()
    print(f"Max probability internal state: {max_prob_state}")
    
    # Get expected internal state
    expected_state = robot.get_expected_internal_state()
    print(f"Expected internal state: {expected_state}")
    
    # Plot belief and reachable sets
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    robot.plot_belief(axes[0])
    robot.plot_reachable_sets(axes[1])
    
    plt.tight_layout()
    plt.show()