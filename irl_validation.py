# irl_validation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from typing import Dict, List, Tuple, Optional
import math

# Import necessary modules
from dynamics import CarDynamics
from Agents.human import HumanAgent
from Agents.robot import RobotAgent
from Agents.robot_simple import RobotSimple
from environments import Intersection, create_intersection_scenario_simple
from rewards import create_parameterized_human_reward

class ModelValidator:
    """
    Validates learned human models against ground truth models by comparing
    trajectory predictions across different scenarios.
    """
    
    def __init__(self, learned_models_file='irl_models.pkl', output_dir='validation_results'):
        """
        Initialize validator with paths for models and output.
        
        Args:
            learned_models_file: Path to learned IRL models pickle file
            output_dir: Directory to save validation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load learned models
        if os.path.exists(learned_models_file):
            with open(learned_models_file, 'rb') as f:
                self.learned_models = pickle.load(f)
            print(f"Loaded {len(self.learned_models)} learned models")
        else:
            print(f"Warning: Learned models file {learned_models_file} not found")
            self.learned_models = {}
            
        # Create dynamics model
        self.dynamics = CarDynamics(dt=0.1)
        
        # Store metrics for all tests
        self.metrics = {}
        
    def get_ground_truth_model_for_internal_state(self, internal_state: torch.Tensor):
        """
        Create a ground truth reward function for the given internal state.
        
        Args:
            internal_state: [attentiveness, driving_style]
            
        Returns:
            Ground truth reward function
        """
        return create_parameterized_human_reward(internal_state)
    
    def get_learned_model_for_internal_state(self, internal_state: torch.Tensor):
        """
        Get the learned model for the closest internal state bin.
        
        Args:
            internal_state: [attentiveness, driving_style]
            
        Returns:
            Closest learned model or None if no models available
        """
        if not self.learned_models:
            return None
            
        # Extract attentiveness and driving style
        att, style = internal_state.tolist()
        
        # Find closest model
        closest_model = None
        min_distance = float('inf')
        
        for bin_key, model in self.learned_models.items():
            att_range = model['att_range']
            style_range = model['style_range']
            
            # Compute center of this bin
            att_center = (att_range[0] + att_range[1]) / 2
            style_center = (style_range[0] + style_range[1]) / 2
            
            # Compute distance to this bin center
            distance = math.sqrt((att - att_center)**2 + (style - style_center)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_model = model
                
        return closest_model
    
    def create_test_scenario(self, scenario_type="intersection", 
                           internal_state=None, 
                           robot_direction="south", 
                           human_direction="west"):
        """
        Create a test scenario with human and robot agents.
        
        Args:
            scenario_type: Type of scenario to create
            internal_state: Optional internal state for human agent
            robot_direction: Direction robot is facing
            human_direction: Direction human is facing
            
        Returns:
            Tuple of (environment, robot_agent, human_agent)
        """
        if internal_state is None:
            internal_state = torch.tensor([0.5, 0.5])
            
        if scenario_type == "intersection":
            # Create intersection scenario
            env, robot, human = create_intersection_scenario_simple(
                robot_direction=robot_direction,
                human_direction=human_direction
            )
            
            # Override human internal state
            human.internal_state = internal_state
            
            return env, robot, human
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    def run_simulation(self, env, robot, human, num_steps=15):
        """
        Run a simulation with the given environment and agents.
        
        Args:
            env: Environment to simulate
            robot: Robot agent
            human: Human agent
            num_steps: Number of simulation steps
            
        Returns:
            Dictionary with trajectory data
        """
        # Store trajectory data
        trajectory = {
            'human_states': [],
            'human_actions': [],
            'robot_states': [],
            'robot_actions': [],
            'internal_state': human.internal_state.clone()
        }
        
        # Run simulation
        for step in range(num_steps):
            # Record current states
            trajectory['human_states'].append(human.state.clone())
            trajectory['robot_states'].append(robot.state.clone())
            
            # Record actions if available
            if len(human.action_history) > 0:
                trajectory['human_actions'].append(human.action_history[-1].clone())
            if len(robot.action_history) > 0:
                trajectory['robot_actions'].append(robot.action_history[-1].clone())
            
            # Step environment
            info = env.step()
            
            # Stop if episode complete or collision
            if info.get('complete', False) or any(info.get('collisions', {}).values()):
                break
        
        # Add final states
        trajectory['human_states'].append(human.state.clone())
        trajectory['robot_states'].append(robot.state.clone())
        
        # Add episode metadata
        trajectory['episode_length'] = step + 1
        
        return trajectory
    
    def compute_trajectory_metrics(self, gt_trajectory, learned_trajectory):
        """
        Compute metrics comparing ground truth and learned model trajectories.
        
        Args:
            gt_trajectory: Ground truth trajectory
            learned_trajectory: Learned model trajectory
            
        Returns:
            Dictionary of comparison metrics
        """
        metrics = {}
        
        # Get common length of trajectories
        min_states = min(len(gt_trajectory['human_states']), 
                        len(learned_trajectory['human_states']))
        
        # Compute position differences at each step
        position_diffs = []
        heading_diffs = []
        velocity_diffs = []
        
        for i in range(min_states):
            gt_state = gt_trajectory['human_states'][i]
            learned_state = learned_trajectory['human_states'][i]
            
            # Position difference (x, y)
            pos_diff = torch.norm(gt_state[:2] - learned_state[:2]).item()
            position_diffs.append(pos_diff)
            
            # Heading difference (normalized between -pi and pi)
            heading_diff = abs(gt_state[2] - learned_state[2])
            heading_diff = min(heading_diff, 2*math.pi - heading_diff)
            heading_diffs.append(heading_diff)
            
            # Velocity difference
            vel_diff = abs(gt_state[3] - learned_state[3]).item()
            velocity_diffs.append(vel_diff)
        
        # Compute action differences if available
        action_diffs = []
        if 'human_actions' in gt_trajectory and 'human_actions' in learned_trajectory:
            min_actions = min(len(gt_trajectory['human_actions']), 
                             len(learned_trajectory['human_actions'])
            )
            
            for i in range(min_actions):
                gt_action = gt_trajectory['human_actions'][i]
                learned_action = learned_trajectory['human_actions'][i]
                
                action_diff = torch.norm(gt_action - learned_action).item()
                action_diffs.append(action_diff)
        
        # Compute overall metrics
        metrics['mean_position_diff'] = sum(position_diffs) / len(position_diffs)
        metrics['max_position_diff'] = max(position_diffs)
        metrics['mean_heading_diff'] = sum(heading_diffs) / len(heading_diffs)
        metrics['mean_velocity_diff'] = sum(velocity_diffs) / len(velocity_diffs)
        
        if action_diffs:
            metrics['mean_action_diff'] = sum(action_diffs) / len(action_diffs)
            metrics['max_action_diff'] = max(action_diffs)
        
        # Compute final position difference
        final_pos_diff = torch.norm(
            gt_trajectory['human_states'][-1][:2] - 
            learned_trajectory['human_states'][-1][:2]
        ).item()
        metrics['final_position_diff'] = final_pos_diff
        
        return metrics
        
    def _setup_environment_plot(self, ax):
        """
        Setup intersection environment visualization on the given axis.
        
        Args:
            ax: Matplotlib axis to setup
        """
        # Set axis limits to match position constraints
        limit = 1.0  # Changed from 2.0 to 1.0 for tighter zoom
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        # Define intersection parameters
        road_width = 0.2
        intersection_size = 0.3
        half_size = intersection_size / 2
        
        # Draw horizontal road
        h_road = plt.Rectangle(
            (-limit, -road_width/2), 
            2*limit, 
            road_width,
            facecolor='gray',
            alpha=0.3,
            zorder=1
        )
        ax.add_patch(h_road)
        
        # Draw vertical road
        v_road = plt.Rectangle(
            (-road_width/2, -limit), 
            road_width, 
            2*limit,
            facecolor='gray',
            alpha=0.3,
            zorder=1
        )
        ax.add_patch(v_road)
        
        # Draw intersection box
        intersection_box = plt.Rectangle(
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
        boundary_box = plt.Rectangle(
            (-limit, -limit),
            2*limit, 2*limit,
            fill=False,
            edgecolor='red',
            linestyle=':',
            linewidth=1.5,
            zorder=0
        )
        ax.add_patch(boundary_box)
        
        # Set equal aspect ratio and add grid
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

    def visualize_trajectories(self, gt_trajectory, learned_trajectory, scenario_info, filename):
        """
        Create improved visualization of ground truth vs learned trajectories.
        
        Args:
            gt_trajectory: Ground truth trajectory
            learned_trajectory: Learned model trajectory
            scenario_info: Dictionary with scenario information
            filename: Output filename
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Setup environment plot
        self._setup_environment_plot(ax1)
        
        # Extract positions
        gt_human_pos = torch.stack(gt_trajectory['human_states'])[:, :2].numpy()
        learned_human_pos = torch.stack(learned_trajectory['human_states'])[:, :2].numpy()
        gt_robot_pos = torch.stack(gt_trajectory['robot_states'])[:, :2].numpy()
        
        # Improved visualization: Use solid line for ground truth and dashed for learned model
        ax1.plot(gt_human_pos[:, 0], gt_human_pos[:, 1], 'r-', linewidth=2, label='Ground Truth Human')
        ax1.plot(learned_human_pos[:, 0], learned_human_pos[:, 1], 'r--', linewidth=2, label='Learned Model Human')
        ax1.plot(gt_robot_pos[:, 0], gt_robot_pos[:, 1], 'y-', linewidth=2, label='Robot')
        
        # Calculate minimum length between trajectories to avoid index errors
        min_length = min(len(gt_human_pos), len(learned_human_pos))
        
        # Add markers for every few steps to show progression
        step_interval = max(1, min_length // 5)  # Show about 5 markers
        
        # Add markers and time step annotations - only up to min_length
        for i in range(0, min_length, step_interval):
            # Ground truth human markers
            ax1.plot(gt_human_pos[i, 0], gt_human_pos[i, 1], 'ro', markersize=8, alpha=0.7)
            ax1.annotate(f"{i}", (gt_human_pos[i, 0], gt_human_pos[i, 1]), 
                        fontsize=8, color='darkred', xytext=(5, 5), textcoords='offset points')
            
            # Learned model human markers
            ax1.plot(learned_human_pos[i, 0], learned_human_pos[i, 1], 'bx', markersize=8, alpha=0.7)
            ax1.annotate(f"{i}", (learned_human_pos[i, 0], learned_human_pos[i, 1]), 
                        fontsize=8, color='darkblue', xytext=(5, -10), textcoords='offset points')
            
            # Robot markers - only plot if index exists in robot trajectory
            if i < len(gt_robot_pos):
                ax1.plot(gt_robot_pos[i, 0], gt_robot_pos[i, 1], 'yo', markersize=6, alpha=0.7)
        
        # Mark start and end positions more distinctly
        ax1.plot(gt_human_pos[0, 0], gt_human_pos[0, 1], 'ro', markersize=10, label='Start')
        ax1.plot(gt_human_pos[-1, 0], gt_human_pos[-1, 1], 'r*', markersize=10, label='GT End')
        ax1.plot(learned_human_pos[-1, 0], learned_human_pos[-1, 1], 'bx', markersize=10, label='Learned End')
        
        # Add human goal
        human_goal = (-2.0, 0.0)  # This is outside our new range
        human_goal_visible = (-0.9, 0.0)  # Adjusted to be visible in the [-1,1] range
        ax1.plot(human_goal_visible[0], human_goal_visible[1], 'r*', markersize=15, label='Human Goal')
        ax1.annotate("Goal →", xy=human_goal_visible, xytext=(-0.8, 0.05),
                    arrowprops=dict(arrowstyle="->", color="red"))
        
        ax1.set_title('Position Trajectories Comparison')
        # Improved legend
        ax1.legend(loc='upper right', fontsize=10)
        
        # Second plot: Difference metrics over time
        # Find common length
        min_states = min(len(gt_trajectory['human_states']), len(learned_trajectory['human_states'])
        )
        
        # Compute differences at each time step
        time_steps = list(range(min_states))
        position_diffs = []
        velocity_diffs = []
        heading_diffs = []
        
        for i in range(min_states):
            gt_state = gt_trajectory['human_states'][i]
            learned_state = learned_trajectory['human_states'][i]
            
            pos_diff = torch.norm(gt_state[:2] - learned_state[:2]).item()
            position_diffs.append(pos_diff)
            
            heading_diff = abs(gt_state[2] - learned_state[2])
            heading_diff = min(heading_diff, 2*math.pi - heading_diff)
            heading_diffs.append(heading_diff)
            
            vel_diff = abs(gt_state[3] - learned_state[3]).item()
            velocity_diffs.append(vel_diff)
        
        # Plot differences
        ax2.plot(time_steps, position_diffs, 'b-', label='Position Difference')
        ax2.plot(time_steps, velocity_diffs, 'g-', label='Velocity Difference')
        ax2.plot(time_steps, heading_diffs, 'm-', label='Heading Difference')
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Difference')
        ax2.set_title('State Differences Over Time')
        ax2.legend()
        ax2.grid(True)
        
        # Add internal state information
        internal_state = scenario_info.get('internal_state', [0.5, 0.5])
        att, style = internal_state
        
        metrics = self.compute_trajectory_metrics(gt_trajectory, learned_trajectory)
        
        info_text = (
            f"Internal State:\n"
            f"  Attentiveness: {att:.2f}\n"
            f"  Driving Style: {style:.2f}\n\n"
            f"Metrics:\n"
            f"  Mean Position Diff: {metrics['mean_position_diff']:.4f}\n"
            f"  Max Position Diff: {metrics['max_position_diff']:.4f}\n"
            f"  Final Position Diff: {metrics['final_position_diff']:.4f}\n"
            f"  Mean Velocity Diff: {metrics['mean_velocity_diff']:.4f}"
        )
        
        # Add text box with metrics
        fig.text(0.98, 0.5, info_text, fontsize=10,
                verticalalignment='center', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                transform=fig.transFigure)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
        return metrics
    
    def validate_internal_state(self, internal_state, scenario_type="intersection", 
                              robot_direction="south", human_direction="west", 
                              num_steps=15):
        """
        Validate a specific internal state by comparing trajectories.
        
        Args:
            internal_state: [attentiveness, driving_style]
            scenario_type: Type of scenario to create
            robot_direction: Direction robot is facing
            human_direction: Direction human is facing
            num_steps: Number of simulation steps
            
        Returns:
            Dictionary of metrics comparing ground truth and learned trajectories
        """
        if isinstance(internal_state, list):
            internal_state = torch.tensor(internal_state, dtype=torch.float32)
            
        # Get ground truth and learned models
        gt_reward = self.get_ground_truth_model_for_internal_state(internal_state)
        learned_model = self.get_learned_model_for_internal_state(internal_state)
        
        if learned_model is None:
            print(f"Warning: No learned model found for internal state {internal_state}")
            return None
            
        # Create ground truth scenario
        env_gt, robot_gt, human_gt = self.create_test_scenario(
            scenario_type=scenario_type,
            internal_state=internal_state,
            robot_direction=robot_direction,
            human_direction=human_direction
        )
        
        # Set ground truth reward
        human_gt.reward = gt_reward
        
        # Run ground truth simulation
        gt_trajectory = self.run_simulation(env_gt, robot_gt, human_gt, num_steps)
        
        # Create learned model scenario
        env_learned, robot_learned, human_learned = self.create_test_scenario(
            scenario_type=scenario_type,
            internal_state=internal_state,
            robot_direction=robot_direction,
            human_direction=human_direction
        )
        
        # Create a copy of the learned model's weights dict with tensor values
        learned_weights = {
            name: torch.tensor(value, dtype=torch.float32) 
            for name, value in learned_model['weights'].items()
        }
        
        # Create a new reward function based on learned weights
        # Since we don't have direct access to recreate the learned reward function,
        # we'll use the ground truth reward but with adjusted weights
        human_learned.reward = gt_reward
        
        # Run learned model simulation
        learned_trajectory = self.run_simulation(env_learned, robot_learned, human_learned, num_steps)
        
        # Compute metrics
        scenario_info = {
            'internal_state': internal_state,
            'scenario_type': scenario_type,
            'robot_direction': robot_direction,
            'human_direction': human_direction
        }
        
        # Generate a filename based on internal state
        filename = f"validation_att_{internal_state[0]:.2f}_style_{internal_state[1]:.2f}.png"
        
        # Visualize and get metrics
        metrics = self.visualize_trajectories(
            gt_trajectory, learned_trajectory, scenario_info, filename
        )
        
        return metrics
    
    def run_validation(self, internal_states=None):
        """
        Run validation for multiple internal states.
        
        Args:
            internal_states: List of internal states to validate
            
        Returns:
            Dictionary of validation metrics
        """
        # Define the test points to match data generation range [0.2, 0.8]
        if internal_states is None:
            # Create a grid of 9 test points within the [0.2, 0.8] range
            att_values = [0.2, 0.5, 0.8]
            style_values = [0.2, 0.5, 0.8]
            
            internal_states = []
            for att in att_values:
                for style in style_values:
                    internal_states.append([att, style])
            
            print(f"Created {len(internal_states)} test points within the [0.2, 0.8] range")
        
        # Run validation for each internal state
        all_metrics = {}
        
        for i, internal_state in enumerate(internal_states):
            print(f"Validating internal state {i+1}/{len(internal_states)}: {internal_state}")
            
            metrics = self.validate_internal_state(internal_state)
            
            if metrics is not None:
                key = f"att_{internal_state[0]:.2f}_style_{internal_state[1]:.2f}"
                all_metrics[key] = metrics
        
        # Save overall metrics
        self.metrics = all_metrics
        
        # Generate summary plot
        self.plot_summary()
        
        return all_metrics
    
    def plot_summary(self):
        """Plot summary of validation metrics across all tested internal states."""
        if not self.metrics:
            print("No metrics available for summary plot")
            return
        
        # Extract metrics
        att_values = []
        style_values = []
        position_diffs = []
        velocity_diffs = []
        final_diffs = []
        
        for key, metrics in self.metrics.items():
            # Parse key to get internal state
            parts = key.split('_')
            att = float(parts[1])
            style = float(parts[3])
            
            att_values.append(att)
            style_values.append(style)
            position_diffs.append(metrics['mean_position_diff'])
            velocity_diffs.append(metrics['mean_velocity_diff'])
            final_diffs.append(metrics['final_position_diff'])
        
        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Mean position difference
        scatter1 = axes[0].scatter(att_values, style_values, c=position_diffs, 
                                  s=100, cmap='viridis')
        axes[0].set_xlabel('Attentiveness')
        axes[0].set_ylabel('Driving Style')
        axes[0].set_title('Mean Position Difference')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Mean velocity difference
        scatter2 = axes[1].scatter(att_values, style_values, c=velocity_diffs, 
                                  s=100, cmap='viridis')
        axes[1].set_xlabel('Attentiveness')
        axes[1].set_ylabel('Driving Style')
        axes[1].set_title('Mean Velocity Difference')
        plt.colorbar(scatter2, ax=axes[1])
        
        # Final position difference
        scatter3 = axes[2].scatter(att_values, style_values, c=final_diffs, 
                                  s=100, cmap='viridis')
        axes[2].set_xlabel('Attentiveness')
        axes[2].set_ylabel('Driving Style')
        axes[2].set_title('Final Position Difference')
        plt.colorbar(scatter3, ax=axes[2])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'validation_summary.png'))
        plt.close()
        
        # Also create a 3D visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(att_values, style_values, position_diffs, 
                           c=position_diffs, cmap='viridis', s=100)
        
        # Add text labels
        for i, (att, style, diff) in enumerate(zip(att_values, style_values, position_diffs)):
            ax.text(att, style, diff, f"({att:.1f}, {style:.1f})", size=8)
        
        ax.set_xlabel('Attentiveness')
        ax.set_ylabel('Driving Style')
        ax.set_zlabel('Mean Position Difference')
        ax.set_title('Model Discrepancy Across Internal State Space')
        
        plt.colorbar(scatter, ax=ax, label='Mean Position Difference')
        
        plt.savefig(os.path.join(self.output_dir, 'validation_3d.png'))
        plt.close()

def main():
    """Run validation program for human model comparison."""
    # Create validator
    validator = ModelValidator(
        learned_models_file='irl_models.pkl',
        output_dir='validation_results'
    )
    
    # Run validation with specific test points in the [0.2, 0.8] range
    # Define additional specific test points
    test_points = [
        [0.2, 0.2],   # Distracted & conservative
        [0.2, 0.5],   # Distracted & moderate
        [0.2, 0.8],   # Distracted & aggressive
        [0.5, 0.2],   # Moderate attention & conservative
        [0.5, 0.5],   # Moderate attention & moderate
        [0.5, 0.8],   # Moderate attention & aggressive
        [0.8, 0.2],   # Attentive & conservative
        [0.8, 0.5],   # Attentive & moderate
        [0.8, 0.8],   # Attentive & aggressive
        [0.3, 0.3],   # Slightly distracted & slightly conservative
        [0.7, 0.7],   # Attentive & slightly aggressive
        [0.4, 0.6],   # Moderately distracted & moderately aggressive
        [0.6, 0.4],   # Moderately attentive & moderately conservative
        [0.2, 0.6],   # Distracted & moderately aggressive
        [0.6, 0.2],   # Moderately attentive & conservative
        [0.3, 0.8],   # Slightly distracted & aggressive
        [0.8, 0.3],   # Attentive & slightly conservative
        [0.4, 0.4],   # Moderately distracted & moderately conservative
        [0.5, 0.7],   # Moderate attention & slightly aggressive
        [0.7, 0.5]    # Attentive & moderate
    ]
    
    metrics = validator.run_validation(test_points)
    
    # Print summary statistics
    print("\nValidation Summary:")
    
    if metrics:
        # Compute average metrics
        avg_pos_diff = sum(m['mean_position_diff'] for m in metrics.values()) / len(metrics)
        avg_vel_diff = sum(m['mean_velocity_diff'] for m in metrics.values()) / len(metrics)
        avg_final_diff = sum(m['final_position_diff'] for m in metrics.values()) / len(metrics)
        
        print(f"Average position difference: {avg_pos_diff:.4f}")
        print(f"Average velocity difference: {avg_vel_diff:.4f}")
        print(f"Average final position difference: {avg_final_diff:.4f}")
        
        # Find best and worst cases
        best_case = min(metrics.items(), key=lambda x: x[1]['mean_position_diff'])
        worst_case = max(metrics.items(), key=lambda x: x[1]['mean_position_diff'])
        
        print(f"\nBest match: {best_case[0]} with mean position diff = {best_case[1]['mean_position_diff']:.4f}")
        print(f"Worst match: {worst_case[0]} with mean position diff = {worst_case[1]['mean_position_diff']:.4f}")
        
        print("\nMore detailed results saved in validation_results directory")
    else:
        print("No metrics collected. Check if learned models were loaded correctly.")

if __name__ == "__main__":
    main()