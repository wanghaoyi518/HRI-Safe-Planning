import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
import sys
import os

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Agents.belief_models import IntervalBelief, Interval
from rewards import create_parameterized_human_reward


class SamplingBasedReachabilityAnalysis:
    """
    Implementation of Algorithm 1: Sampling-based Reachability Analysis (SRA)
    from the paper "Safe Robot Planning Through Understanding Human Behaviors"
    """
    
    def __init__(self, dynamics, human_model):
        """
        Initialize with dynamics model and human model.
        
        Args:
            dynamics: System dynamics model
            human_model: Human agent model with IRL-learned reward
        """
        self.dynamics = dynamics
        self.human_model = human_model
        
    def compute_reachable_sets(self, 
                             current_human_state: torch.Tensor,
                             current_robot_state: torch.Tensor,
                             belief_interval: Interval,
                             num_samples: int, 
                             robot_controls: torch.Tensor, 
                             time_horizon: int) -> List[torch.Tensor]:
        """
        Compute reachable sets via sampling-based approach (Algorithm 1).
        
        Args:
            current_human_state: Current state of human agent
            current_robot_state: Current state of robot agent
            belief_interval: Interval of human internal states to sample from
            num_samples: Number of samples to use
            robot_controls: Sequence of planned robot controls
            time_horizon: Number of steps to look ahead
            
        Returns:
            List of reachable sets (as tensors) for each time step
        """
        # 1. Sample internal states from the belief interval
        internal_state_samples = belief_interval.uniform_sample(num_samples)
        
        # Initialize trajectories for each sampled internal state
        all_human_trajectories = []
        
        # 2. For each sampled phi
        for phi in internal_state_samples:
            # Initialize trajectory with current human state
            human_trajectory = [current_human_state.clone()]
            human_state = current_human_state.clone()
            robot_state = current_robot_state.clone()
            
            # Store original internal state
            original_internal_state = self.human_model.internal_state.clone()
            
            # Set human's internal state to the sampled phi
            self.human_model.internal_state = phi
            
            # Use IRL-learned reward model based on this internal state
            # This is critical for accurate human action prediction
            original_reward = self.human_model.reward
            self.human_model.reward = create_parameterized_human_reward(phi)
            
            # Propagate dynamics for time horizon
            for t in range(time_horizon):
                # Get robot control for this time step
                if t < len(robot_controls):
                    robot_control = robot_controls[t]
                else:
                    # Use last control if horizon exceeds provided controls
                    robot_control = robot_controls[-1]
                
                # Apply robot action to robot state
                next_robot_state = self.dynamics(robot_state, robot_control)
                
                # Create environment state for human model
                env_state = {"robot": {"state": next_robot_state, "action": robot_control}}
                
                # Compute human action using IRL-learned reward
                human_action = self.human_model.compute_control(t, human_state, env_state)
                
                # Apply human action to human state
                next_human_state = self.dynamics(human_state, human_action)
                
                # Add to human trajectory
                human_trajectory.append(next_human_state)
                
                # Update states for next iteration
                human_state = next_human_state
                robot_state = next_robot_state
            
            # Restore original internal state and reward
            self.human_model.internal_state = original_internal_state
            self.human_model.reward = original_reward
            
            # Add completed human trajectory to collection
            all_human_trajectories.append(human_trajectory)
        
        # 3. Compute reachable sets at each time step
        reachable_sets = []
        
        for t in range(time_horizon + 1):
            # Collect human states at time t from all trajectories
            human_states_at_t = [traj[t] for traj in all_human_trajectories]
            
            # Stack into a tensor
            human_states_tensor = torch.stack(human_states_at_t)
            
            # Add to reachable sets
            reachable_sets.append(human_states_tensor)
        
        return reachable_sets
    
    def compute_reachable_sets_with_belief(self, 
                                        current_human_state: torch.Tensor,
                                        current_robot_state: torch.Tensor,
                                        belief: IntervalBelief,
                                        robot_controls: torch.Tensor,
                                        time_horizon: int,
                                        total_samples: int) -> List[torch.Tensor]:
        """
        Compute reachable sets using adaptive sampling based on belief distribution.
        
        Args:
            current_human_state: Current state of human agent
            current_robot_state: Current state of robot agent
            belief: Belief distribution over human internal states
            robot_controls: Sequence of planned robot controls
            time_horizon: Number of steps to look ahead
            total_samples: Total number of samples to use
            
        Returns:
            List of reachable sets for each time step
        """
        # Log some diagnostic information
        #print(f"Computing reachable sets with {len(belief.intervals)} intervals and {total_samples} total samples")
        
        # Allocate samples based on belief - this is the key function we modified
        sample_allocation = belief.sample_proportionally(total_samples)
        
        # Log how samples are allocated
        #print(f"Sample allocation: {sample_allocation}")
        
        # Initialize empty reachable sets for each time step
        all_states = [[] for _ in range(time_horizon + 1)]
        
        # Perform reachability analysis for each interval with samples allocated
        for interval_idx, num_samples in sample_allocation.items():
            # Skip intervals with no samples
            if num_samples <= 0:
                continue
                
            interval = belief.intervals[interval_idx]
            
            # Log which interval we're processing and how many samples
            # Print(f"Processing interval {interval} with {num_samples} samples")
            
            # Compute reachable sets for this interval
            interval_reachable_sets = self.compute_reachable_sets(
                current_human_state,
                current_robot_state,
                interval,
                num_samples,
                robot_controls,
                time_horizon
            )
            
            # Add states to combined reachable sets
            for t in range(time_horizon + 1):
                all_states[t].append(interval_reachable_sets[t])
        
        # Combine reachable sets for each time step
        combined_reachable_sets = []
        
        for t in range(time_horizon + 1):
            if all_states[t]:
                # Concatenate all states at this time step
                combined = torch.cat(all_states[t], dim=0)
                combined_reachable_sets.append(combined)
            else:
                # If no states for this time step, create a placeholder
                placeholder = torch.zeros(1, current_human_state.shape[0])
                combined_reachable_sets.append(placeholder)
        
        return combined_reachable_sets

    def compute_convex_hull(self, states: torch.Tensor) -> np.ndarray:
        """
        Compute convex hull of a set of states.
        
        Args:
            states: Tensor of shape [n_states, state_dim]
            
        Returns:
            Hull vertices as numpy array
        """
        # Convert to numpy for convex hull computation
        points = states.detach().cpu().numpy()
        
        # Extract position coordinates
        positions = points[:, :2]  # Assuming first two dimensions are x, y positions
        
        try:
            # Only compute hull if we have enough points
            if len(positions) >= 3:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(positions)
                hull_vertices = positions[hull.vertices]
                return hull_vertices
            else:
                return positions
        except:
            # Handle case where points are collinear
            return positions
    
    def check_safety_constraint(self, 
                              robot_state: torch.Tensor, 
                              human_reachable_set: torch.Tensor,
                              safety_distance: float) -> bool:
        """
        Check if robot state satisfies safety constraints with respect to human reachable set.
        
        Args:
            robot_state: Robot state to check
            human_reachable_set: Set of possible human states
            safety_distance: Minimum safe distance
            
        Returns:
            True if state is safe
        """
        # Extract positions
        robot_position = robot_state[:2]
        human_positions = human_reachable_set[:, :2]
        
        # Compute distances
        distances = torch.norm(human_positions - robot_position.unsqueeze(0), dim=1)
        
        # Check if minimum distance exceeds safety threshold
        return torch.min(distances) >= safety_distance
    
    def plot_reachable_sets(self, 
                          reachable_sets: List[torch.Tensor],
                          robot_state: torch.Tensor = None,
                          human_state: torch.Tensor = None,
                          ax=None):
        """
        Plot reachable sets for visualization.
        
        Args:
            reachable_sets: List of human reachable sets for each time step
            robot_state: Current robot state (optional)
            human_state: Current human state (optional)
            ax: Matplotlib axis (optional)
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Configure visualization to match the intersection scenario
        self._setup_intersection_environment(ax)
        
        # Plot reachable sets with different colors for each time step
        colors = ['lightblue', 'skyblue', 'blue', 'darkblue', 'purple', 'red']
        
        # Increase marker size and alpha for better visibility
        marker_size = 30
        scatter_alpha = 0.7
        
        for t, reachable_set in enumerate(reachable_sets):
            if t >= len(colors):
                break
            
            # Extract positions for visualization
            positions = reachable_set[:, :2].detach().cpu().numpy()
            
            # Skip if no positions
            if len(positions) == 0:
                continue
            
            # Plot scatter points with larger markers and higher opacity
            ax.scatter(positions[:, 0], positions[:, 1], 
                      c=colors[t], alpha=scatter_alpha, s=marker_size, 
                      label=f'Human t+{t}' if t > 0 else 'Human t',
                      edgecolors='black')
            
            # Add convex hull with higher opacity
            hull_vertices = self.compute_convex_hull(reachable_set)
            if len(hull_vertices) >= 3:
                polygon = Polygon(hull_vertices, alpha=0.4, facecolor=colors[t], edgecolor='black', linewidth=1.5)
                ax.add_patch(polygon)
        
        # Plot current human position if provided - use circle marker instead of rectangle
        if human_state is not None:
            human_position = human_state[:2].detach().cpu().numpy()
            ax.plot(human_position[0], human_position[1], 'ro', markersize=15, label='Human')
        
        # Plot robot position if provided - use circle marker instead of rectangle
        if robot_state is not None:
            robot_position = robot_state[:2].detach().cpu().numpy()
            ax.plot(robot_position[0], robot_position[1], 'yo', markersize=15, label='Robot')
        
        # Set labels and legend
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Human Reachable Sets with Robot Position')
        
        # Add legend with better positioning
        ax.legend(loc='upper right', fontsize=10)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        return ax
    
    def _setup_intersection_environment(self, ax):
        """
        Setup visualization environment for intersection scenario.
        
        Args:
            ax: Matplotlib axis
        """
        # Set axis limits to match position constraints
        limit = 2.0
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        # Define intersection parameters (match exactly with data_generation.py)
        road_width = 0.2
        intersection_size = 0.3
        
        # Draw horizontal road - full width with correct positioning
        h_road = Rectangle(
            (-limit, -road_width/2), 
            2*limit, 
            road_width,
            facecolor='gray',
            alpha=0.3,
            zorder=1
        )
        ax.add_patch(h_road)
        
        # Draw vertical road - full height with correct positioning
        v_road = Rectangle(
            (-road_width/2, -limit), 
            road_width, 
            2*limit,
            facecolor='gray',
            alpha=0.3,
            zorder=1
        )
        ax.add_patch(v_road)
        
        # Define intersection boundaries
        half_size = intersection_size / 2
        
        # Draw intersection box
        intersection_box = Rectangle(
            (-half_size, -half_size), 
            intersection_size, 
            intersection_size,
            facecolor='gray',
            alpha=0.1,
            edgecolor='white',
            linestyle=':',
            zorder=2
        )
        ax.add_patch(intersection_box)
        
        # Draw boundary box to show position constraints
        boundary_box = Rectangle(
            (-2.0, -2.0),
            4.0, 4.0,
            fill=False,
            edgecolor='red',
            linestyle=':',
            linewidth=1.5,
            zorder=0
        )
        ax.add_patch(boundary_box)
        
        # Draw human goal
        human_goal = (-2.0, 0.0)
        ax.plot(human_goal[0], human_goal[1], 'r*', markersize=15, label='Human Goal')
        ax.add_patch(Circle(human_goal, 0.1, color='red', alpha=0.3))


class AdaptiveIntervalPlanner:
    """
    Implementation of Algorithm 2: Safe Planning with Adaptive Interval-Based Reachability Analysis
    """
    
    def __init__(self, 
                dynamics, 
                human_model, 
                robot_reward_fn, 
                safety_distance: float = 0.2):
        """
        Initialize planner with dynamics, human model, and safety parameters.
        
        Args:
            dynamics: System dynamics model
            human_model: Human agent model with IRL-learned reward
            robot_reward_fn: Robot's reward function
            safety_distance: Minimum safe distance to maintain
        """
        self.dynamics = dynamics
        self.human_model = human_model
        self.robot_reward = robot_reward_fn
        self.safety_distance = safety_distance
        self.reachability_analyzer = SamplingBasedReachabilityAnalysis(dynamics, human_model)
        
    def plan(self, 
            current_robot_state: torch.Tensor,
            current_human_state: torch.Tensor,
            belief: IntervalBelief,
            planning_horizon: int = 5,
            total_samples: int = 100,
            refinement_threshold: float = 0.3,
            optimization_steps: int = 20) -> torch.Tensor:
        """
        Plan safe robot trajectory using adaptive sampling (Algorithm 2).
        
        Args:
            current_robot_state: Current robot state
            current_human_state: Current human state
            belief: Current belief over human internal states
            planning_horizon: Number of steps to look ahead
            total_samples: Total number of samples to use
            refinement_threshold: Threshold for interval refinement
            optimization_steps: Number of steps for trajectory optimization
            
        Returns:
            Planned robot control action
        """
        # 1. Allocate samples based on belief
        sample_allocation = belief.sample_proportionally(total_samples)
        
        # 2. Initialize empty reachable sets for each time step
        reachable_sets = [[] for _ in range(planning_horizon + 1)]
        
        # Initial robot control sequence for reachability analysis
        initial_robot_controls = torch.zeros(planning_horizon, self.dynamics.nu)
        
        # 3. Perform reachability analysis for each interval
        for interval_idx, num_samples in sample_allocation.items():
            if num_samples <= 0:
                continue
                
            interval = belief.intervals[interval_idx]
            
            # Compute reachable sets for this interval
            interval_reachable_sets = self.reachability_analyzer.compute_reachable_sets(
                current_human_state,
                current_robot_state,
                interval,
                num_samples,
                initial_robot_controls,
                planning_horizon
            )
            
            # Add states to combined reachable sets
            for t in range(planning_horizon + 1):
                reachable_sets[t].append(interval_reachable_sets[t])
        
        # 4. Combine reachable sets for each time step
        combined_reachable_sets = []
        
        for t in range(planning_horizon + 1):
            if reachable_sets[t]:
                # Concatenate all states at this time step
                combined = torch.cat(reachable_sets[t], dim=0)
                combined_reachable_sets.append(combined)
            else:
                # If no states for this time step, create a placeholder
                combined_reachable_sets.append(torch.zeros(1, current_human_state.shape[0]))
        
        # 5. Plan safe robot trajectory using reachable sets
        best_action = self.optimize_trajectory(
            current_robot_state,
            combined_reachable_sets,
            planning_horizon,
            optimization_steps
        )
        
        return best_action
    
    def optimize_trajectory(self,
                          current_state: torch.Tensor,
                          reachable_sets: List[torch.Tensor],
                          planning_horizon: int,
                          optimization_steps: int) -> torch.Tensor:
        """
        Optimize robot trajectory given reachable sets.
        
        Args:
            current_state: Current robot state
            reachable_sets: Human reachable sets
            planning_horizon: Planning horizon
            optimization_steps: Number of optimization steps
            
        Returns:
            Optimized robot control action
        """
        # Initialize planning
        current_action = torch.zeros(self.dynamics.nu)
        best_action = current_action.clone()
        best_reward = float('-inf')
        
        # Optimization using random sampling with safety constraints
        for _ in range(optimization_steps):
            # Sample a random action with noise around current best action
            action_noise = torch.randn_like(best_action) * 0.2
            trial_action = best_action + action_noise
            
            # Clamp to valid control range
            trial_action = torch.clamp(trial_action, -1.0, 1.0)
            
            # Simulate trajectory for planning horizon
            simulated_state = current_state.clone()
            simulated_states = [simulated_state.clone()]
            simulated_actions = [trial_action.clone()]
            
            # Check if trajectory is safe
            is_safe = True
            
            for h in range(min(planning_horizon, len(reachable_sets) - 1)):
                # Propagate state using dynamics
                simulated_state = self.dynamics(simulated_state, trial_action)
                
                # Store for trajectory
                simulated_states.append(simulated_state.clone())
                simulated_actions.append(trial_action.clone())
                
                # Check safety constraint
                if not self.reachability_analyzer.check_safety_constraint(
                    simulated_state, reachable_sets[h + 1], self.safety_distance
                ):
                    is_safe = False
                    break
            
            # Skip unsafe trajectories
            if not is_safe:
                continue
            
            # Compute total reward
            total_reward = torch.tensor(0.0, dtype=torch.float32)
            
            for h in range(len(simulated_states) - 1):
                # Compute reward for this state-action pair
                step_reward = self.robot_reward(h, simulated_states[h], simulated_actions[h])
                total_reward += step_reward
            
            # Update best action if we found better reward
            if total_reward > best_reward:
                best_reward = total_reward
                best_action = trial_action.clone()
        
        return best_action
    
    def plot_planning_results(self,
                            robot_state: torch.Tensor,
                            human_state: torch.Tensor,
                            reachable_sets: List[torch.Tensor],
                            planned_action: torch.Tensor,
                            ax=None):
        """
        Plot planning results including robot, human, and reachable sets.
        
        Args:
            robot_state: Current robot state
            human_state: Current human state
            reachable_sets: Human reachable sets
            planned_action: Planned robot action
            ax: Matplotlib axis (optional)
            
        Returns:
            Matplotlib axis
        """
        # Use reachability analyzer to plot reachable sets
        ax = self.reachability_analyzer.plot_reachable_sets(
            reachable_sets, robot_state, human_state, ax
        )
        
        # Add planned action visualization
        # Create a small arrow showing the planned action direction
        robot_pos = robot_state[:2].detach().cpu().numpy()
        arrow_scale = 0.2
        dx = planned_action[1].item() * arrow_scale  # Using acceleration for length
        dy = planned_action[0].item() * arrow_scale  # Using steering for direction
        
        ax.arrow(robot_pos[0], robot_pos[1], dx, dy,
                head_width=0.05, head_length=0.05, fc='black', ec='black',
                zorder=12, label='Planned Action')
        
        # Update title
        ax.set_title(f'Planning Results - Action: [{planned_action[0]:.2f}, {planned_action[1]:.2f}]')
        
        return ax


if __name__ == "__main__":
    # Test the SamplingBasedReachabilityAnalysis with intersection environment
    from dynamics import CarDynamics
    from Agents.human import HumanAgent
    from rewards import create_parameterized_human_reward, create_robot_reward
    
    # Create dynamics model
    dynamics = CarDynamics(dt=0.1)
    
    # Create intersection scenario similar to data_generation.py
    # Define directions and positions
    robot_direction = "south"  # Robot comes from south
    human_direction = "west"   # Human comes from west
    
    directions = {
        "north": (0.0, -0.5, np.pi/2),    # (x, y, theta)
        "south": (0.0, 0.5, -np.pi/2),
        "east": (-0.5, 0.0, 0.0),
        "west": (0.5, 0.0, np.pi)
    }
    
    # Setup robot and human initial states
    robot_x, robot_y, robot_theta = directions[robot_direction]
    robot_init_state = torch.tensor([robot_x, robot_y, robot_theta, 0.1])  # Small initial velocity
    
    human_x, human_y, human_theta = directions[human_direction]
    human_init_state = torch.tensor([human_x, human_y, human_theta, 0.1])  # Small initial velocity
    
    # Create human agent with a specific internal state
    true_internal_state = torch.tensor([0.8, 0.7])  # High attentiveness, moderately aggressive
    human = HumanAgent(
        dynamics,
        human_init_state,
        internal_state=true_internal_state,
        reward=create_parameterized_human_reward(true_internal_state),
        name="human",
        color="red"
    )
    
    # Create reachability analyzer
    reachability_analyzer = SamplingBasedReachabilityAnalysis(dynamics, human)
    
    # Create belief over human internal state
    belief = IntervalBelief()
    
    # Update belief to create non-uniform distribution
    def test_likelihood(phi):
        att, style = phi
        target_att = 0.7  # Close to true value but not exact
        target_style = 0.6
        dist = torch.sqrt((att - target_att)**2 + (style - target_style)**2)
        return torch.exp(-5.0 * dist)
    
    # Update and refine belief multiple times to create a more specific distribution
    for _ in range(3):
        belief.update(test_likelihood)
        belief.refine(threshold=0.5)
    
    # Create robot control sequence for testing - use varied controls to generate diverse reachable sets
    robot_controls = torch.tensor([
        [0.0, 0.2],  # Accelerate forward
        [0.0, 0.3], 
        [0.1, 0.3],  # Turn slightly while accelerating
        [0.1, 0.2],
        [0.0, 0.1]   # Slow down
    ])
    
    # Compute reachable sets with more samples
    reachable_sets = reachability_analyzer.compute_reachable_sets_with_belief(
        human_init_state,
        robot_init_state,
        belief,
        robot_controls,
        time_horizon=5,
        total_samples=200  # Use more samples for better visualization
    )
    
    # Plot reachable sets
    plt.figure(figsize=(10, 10))
    reachability_analyzer.plot_reachable_sets(
        reachable_sets, 
        robot_state=robot_init_state, 
        human_state=human_init_state
    )
    plt.savefig("human_reachable_sets.png")
    plt.close()
    
    # Create robot reward function
    def robot_reward(t, x, u):
        # Goal: cross intersection safely
        goal_pos = torch.tensor([0.0, -1.0])  # Goal at bottom of intersection
        current_pos = x[:2]
        
        # Distance to goal
        dist_to_goal = torch.norm(current_pos - goal_pos)
        
        # Control cost
        control_cost = torch.sum(u**2)
        
        # Combined reward
        return -dist_to_goal - 0.1 * control_cost
    
    # Create adaptive planner
    planner = AdaptiveIntervalPlanner(
        dynamics,
        human,
        robot_reward,
        safety_distance=0.2
    )
    
    # Plan action
    planned_action = planner.plan(
        robot_init_state,
        human_init_state,
        belief,
        planning_horizon=5,
        total_samples=200,
        refinement_threshold=0.3,
        optimization_steps=20
    )
    
    print(f"Planned robot action: {planned_action}")
    
    # Visualize planning results
    plt.figure(figsize=(10, 10))
    planner.plot_planning_results(
        robot_init_state,
        human_init_state,
        reachable_sets,
        planned_action
    )
    plt.savefig("adaptive_planning_results.png")
    plt.close()
    
    # Also plot the belief for reference
    plt.figure(figsize=(8, 6))
    belief.plot(show_center=True)
    plt.title("Belief over Human Internal State")
    plt.savefig("belief_distribution.png")
    plt.close()
    
    print("Visualization completed. Check the generated PNG files.")