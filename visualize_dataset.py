import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple
import os

def load_dataset(filepath: str):
    """Load dataset from pickle file."""
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def visualize_trajectory(trajectory: Dict, index: int):
    """Visualize a single trajectory with spatial paths."""
    # Extract data
    human_states = torch.stack(trajectory['human_states'])
    robot_states = torch.stack(trajectory['robot_states'])
    internal_state = trajectory['internal_state']
    scenario_type = trajectory['scenario_type']
    
    # Extract position data for plotting
    human_positions = human_states[:, :2].numpy()
    robot_positions = robot_states[:, :2].numpy()
    
    # Create trajectory plot with custom layout for legend
    fig = plt.figure(figsize=(12, 8))
    
    # Create a gridspec with space for the legends - trajectory will take 70% of width
    # The remaining 30% is split between upper legend (10%) and lower info box (20%)
    gs = fig.add_gridspec(2, 2, width_ratios=[7, 3], height_ratios=[1, 1])
    
    # Create axes
    ax_traj = fig.add_subplot(gs[:, 0])       # Left side (full height) for trajectory plot
    ax_legend = fig.add_subplot(gs[0, 1])     # Upper right for legend
    ax_info = fig.add_subplot(gs[1, 1])       # Lower right for sample info
    
    # Turn off axes for legend and info
    ax_legend.axis('off')
    ax_info.axis('off')
    
    # Plot paths as lines
    line_human = ax_traj.plot(human_positions[:, 0], human_positions[:, 1], 'r-', linewidth=1.5, alpha=0.7, label='Human')
    line_robot = ax_traj.plot(robot_positions[:, 0], robot_positions[:, 1], 'y-', linewidth=1.5, alpha=0.7, label='Robot')
    
    # Add human goal
    human_goal = (-2.0, 0.0)
    goal_human = ax_traj.plot(human_goal[0], human_goal[1], 'r*', markersize=15, label='Human Goal')
    ax_traj.add_patch(plt.Circle(human_goal, 0.1, color='red', alpha=0.3))
    
    # Add dots for each time step
    human_dots = ax_traj.scatter(human_positions[:, 0], human_positions[:, 1], c='red', s=20, zorder=5, label='Human Position')
    robot_dots = ax_traj.scatter(robot_positions[:, 0], robot_positions[:, 1], c='yellow', s=20, zorder=5, label='Robot Position')
    
    # Add time step numbers at regular intervals
    step_interval = max(1, len(human_positions) // 5)  # Show ~5 time labels
    for i in range(0, len(human_positions), step_interval):
        ax_traj.annotate(f"{i}", (human_positions[i, 0], human_positions[i, 1]), 
                        fontsize=8, ha='right', va='bottom', color='darkred')
        ax_traj.annotate(f"{i}", (robot_positions[i, 0], robot_positions[i, 1]), 
                        fontsize=8, ha='left', va='bottom', color='darkgoldenrod')
    
    # Mark start and end points
    start_human = ax_traj.plot(human_positions[0, 0], human_positions[0, 1], 'ro', markersize=10, label='Human Start')
    end_human = ax_traj.plot(human_positions[-1, 0], human_positions[-1, 1], 'rx', markersize=10, label='Human End')
    start_robot = ax_traj.plot(robot_positions[0, 0], robot_positions[0, 1], 'yo', markersize=10, label='Robot Start')
    end_robot = ax_traj.plot(robot_positions[-1, 0], robot_positions[-1, 1], 'yx', markersize=10, label='Robot End')
    
    # Add environment visualization based on scenario type
    # if scenario_type == 'highway':
    #     # Simple highway visualization
    #     highway_width = 0.3
    #     lane_center = 0.0
    #     ax_traj.axhline(y=lane_center, color='gray', linestyle='-', alpha=0.5)
    #     ax_traj.axhline(y=lane_center + 0.13, color='gray', linestyle='--', alpha=0.5)
    #     ax_traj.axhline(y=lane_center - 0.13, color='gray', linestyle='--', alpha=0.5)
    # elif scenario_type == 'intersection':
    #     # Simple intersection visualization
    #     intersection_size = 0.3
    #     road_width = 0.2
        
    #     # Draw horizontal road
    #     ax_traj.add_patch(plt.Rectangle((-2, -road_width/2), 4, road_width, 
    #                                    color='gray', alpha=0.3))
    #     # Draw vertical road
    #     ax_traj.add_patch(plt.Rectangle((-road_width/2, -2), road_width, 4, 
    #                                    color='gray', alpha=0.3))
    

        #     # Simple intersection visualization
    intersection_size = 0.3
    road_width = 0.2

    # Draw horizontal road
    ax_traj.add_patch(plt.Rectangle((-2, -road_width/2), 4, road_width, 
                                   color='gray', alpha=0.3))
    # Draw vertical road
    ax_traj.add_patch(plt.Rectangle((-road_width/2, -2), road_width, 4, 
                                   color='gray', alpha=0.3))
    
    # Set visualization limits to match position constraints
    ax_traj.set_xlim(-2.0, 2.0)
    ax_traj.set_ylim(-2.0, 2.0)
    
    # Draw boundary box to show position constraints
    constraint_box = plt.Rectangle((-2.0, -2.0), 4.0, 4.0,
                                  fill=False, 
                                  edgecolor='red', 
                                  linestyle=':', 
                                  linewidth=1.5)
    ax_traj.add_patch(constraint_box)
    
    # Set equal aspect ratio and add grid
    ax_traj.set_aspect('equal')
    ax_traj.grid(True, linestyle='--', alpha=0.7)
    ax_traj.set_xlabel('X Position')
    ax_traj.set_ylabel('Y Position')
    ax_traj.set_title(f"Agent Trajectories - {scenario_type.capitalize()} Scenario")
    
    # Create trajectory legend in the upper right area
    # Combine all plot handles for the legend
    legend_elements = line_human + line_robot + goal_human + start_human + end_human + start_robot + end_robot
    legend_labels = [h.get_label() for h in legend_elements]
    
    # Create the legend in upper right
    ax_legend.legend(legend_elements, legend_labels, loc='center', fontsize=10)
    
    # Add trajectory information as text in the lower right
    attentiveness, driving_style = internal_state.numpy()
    info_text = (
        f"Sample {index+1}\n\n"
        f"Scenario: {scenario_type}\n\n"
        f"Human Internal State:\n"
        f"  Attentiveness: {attentiveness:.2f}\n"
        f"  Driving Style: {driving_style:.2f}\n\n"
        f"Episode Length: {trajectory['episode_length']}"
    )
    
    # Add a nice box around the information
    ax_info.text(0.5, 0.5, info_text, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                transform=ax_info.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray'))
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def visualize_samples(dataset, num_samples=None):
    """Visualize trajectories representing each unique internal state combination."""
    # Create output directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    
    # If num_samples is None, visualize all trajectories
    if num_samples is None:
        trajectories_to_visualize = dataset
    # Otherwise limit to the specified number of samples
    else:
        trajectories_to_visualize = dataset[:num_samples]
    
    # Visualize each trajectory
    for idx, sample in enumerate(trajectories_to_visualize):
        # Get the exact internal state values
        att, style = sample['internal_state'].numpy()
        
        # Create visualization
        fig = visualize_trajectory(sample, idx)
        
        # Save the figure with descriptive filename using exact values
        filename = f"visualizations/att_{att:.2f}_style_{style:.2f}_{sample['scenario_type']}.png"
        fig.savefig(filename)
        plt.close(fig)
        
        print(f"Saved trajectory visualization to {filename}")
    
    print(f"Visualized {len(trajectories_to_visualize)} trajectories")

def main():
    """Main function to load and visualize dataset."""
    dataset_path = "data/irl_dataset.pkl"
    
    # Check if dataset file exists
    if not os.path.exists(dataset_path):
        print(f"Dataset file {dataset_path} not found. Please run data_generation.py first.")
        return
    
    # Load dataset
    dataset = load_dataset(dataset_path)
    print(f"Loaded dataset with {len(dataset)} trajectories")
    
    # Visualize samples - pass None to show all unique combinations
    visualize_samples(dataset, num_samples=None)
    print("Visualization complete. Images saved to visualizations/ directory.")
    
    
    # Show a summary of the dataset
    scenario_counts = {}
    attentiveness_values = []
    driving_style_values = []
    
    for traj in dataset:
        scenario = traj['scenario_type']
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        internal_state = traj['internal_state']
        attentiveness_values.append(internal_state[0].item())
        driving_style_values.append(internal_state[1].item())
    
    print("\nDataset Summary:")
    print(f"Total traj.: {len(dataset)}")
    print(f"Scenario dist.: {scenario_counts}")
    print(f"Att. range: [{min(attentiveness_values):.2f}, {max(attentiveness_values):.2f}]")
    print(f"Style range: [{min(driving_style_values):.2f}, {max(driving_style_values):.2f}]")

if __name__ == "__main__":
    main()
