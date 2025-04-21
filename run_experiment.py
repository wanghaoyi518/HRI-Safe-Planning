import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import os
from typing import Dict, List, Tuple, Optional
import time

from dynamics import CarDynamics
from Agents.human import HumanAgent
from Agents.robot import RobotAgent
from Agents.belief_models import IntervalBelief
from reachability import SamplingBasedReachabilityAnalysis
from environments import Intersection
from rewards import create_robot_reward, create_parameterized_human_reward


def create_intersection_scenario(attentiveness=0.8, driving_style=0.7):
    """
    Create an intersection scenario with robot and human agents.
    
    Args:
        attentiveness: Human attentiveness level
        driving_style: Human driving style (aggressiveness)
        
    Returns:
        Tuple of (environment, robot_agent, human_agent)
    """
    # Create dynamics model
    dynamics = CarDynamics(dt=0.1)
    
    # Create environment
    env = Intersection(road_width=0.2, intersection_size=0.3)
    
    # Create robot agent at south position, facing north
    robot_init_state = torch.tensor([0.0, 0.5, -np.pi/2, 0.0])
    robot_reward_fn = create_robot_reward(
        collision_weight=50.0,
        goal_weight=10.0,
        info_gain_weight=2.0,
        goal_position=torch.tensor([0.0, -2.0])  # Goal at bottom
    )
    
    robot = RobotAgent(
        dynamics,
        robot_init_state,
        reward=robot_reward_fn,
        name="robot",
        color="yellow",
        planning_horizon=5,
        optimization_steps=20,
        info_gain_weight=2.0,
        safety_distance=0.2
    )
    
    # Create human agent at west position, facing east
    human_init_state = torch.tensor([0.5, 0.0, np.pi, 0.0])
    internal_state = torch.tensor([attentiveness, driving_style])
    human_reward_fn = create_parameterized_human_reward(internal_state)
    
    human = HumanAgent(
        dynamics,
        human_init_state,
        internal_state=internal_state,
        reward=human_reward_fn,
        name="human",
        color="red",
        planning_horizon=4
    )
    
    # Register agents with environment
    env.register_agent(robot)
    env.register_agent(human)
    
    # Register human with robot
    robot.register_human(human)
    
    # Create reachability analyzer
    reachability_analyzer = SamplingBasedReachabilityAnalysis(dynamics, human)
    robot.reachability_analyzer = reachability_analyzer
    
    return env, robot, human


def run_experiment(
    num_steps: int = 15,
    attentiveness: float = 0.8,
    driving_style: float = 0.7,
    visualization_dir: str = "visualizations"
):
    """
    Run the experiment with active information gathering and safe planning.
    
    Args:
        num_steps: Number of simulation steps
        attentiveness: True human attentiveness (0-1)
        driving_style: True human driving style (0-1)
        visualization_dir: Directory to save visualization images
        
    Returns:
        Dictionary with experiment data
    """
    print(f"Running experiment with attentiveness={attentiveness}, driving_style={driving_style}")
    
    # Create directory for visualizations
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Create scenario
    env, robot, human = create_intersection_scenario(attentiveness, driving_style)
    
    # Data collection
    experiment_data = {
        'human_states': [],
        'robot_states': [],
        'human_actions': [],
        'robot_actions': [],
        'reachable_sets': [],
        'belief_states': [],
        'entropy_history': [],
        'expected_values': [],
        'actual_internal_state': human.internal_state.clone()
    }
    
    # Simulation loop
    for t in range(num_steps):
        print(f"Simulation step {t+1}/{num_steps}")
        
        # Store current states
        experiment_data['human_states'].append(human.state.clone())
        experiment_data['robot_states'].append(robot.state.clone())
        
        # Store belief state
        experiment_data['belief_states'].append({
            'intervals': robot.belief.intervals.copy(),
            'probs': robot.belief.probs.clone()
        })
        
        # Compute reachable sets using SamplingBasedReachabilityAnalysis
        reachable_sets = robot.compute_reachable_sets(
            "human",
            time_horizon=5,
            num_samples=30
        )
        experiment_data['reachable_sets'].append(reachable_sets)
        
        # Store entropy and expected value
        entropy = robot.belief.entropy()
        expected = robot.belief.expected_value()
        experiment_data['entropy_history'].append(entropy.item())
        experiment_data['expected_values'].append(expected.clone())
        
        # Create environment state
        env_state = {
            "human": human
        }
        
        # Compute robot action using active information gathering
        robot_action = robot.compute_control(t, robot.state, env_state)
        
        # Set robot action
        robot.set_action(robot_action)
        experiment_data['robot_actions'].append(robot_action.clone())
        
        # Compute human action
        human_action = human.compute_control(t, human.state, {"robot": robot})
        experiment_data['human_actions'].append(human_action.clone())
        
        # Execute environment step
        info = env.step()
        
        # Update robot belief based on observed human action
        robot.update_belief("human", human_action)
        
        # Check termination conditions
        if info.get('complete', False) or any(info.get('collisions', {}).values()):
            print(f"Simulation completed at step {t+1}")
            break
    
    # Create visualizations
    visualize_trajectories(experiment_data, visualization_dir)
    visualize_reachable_sets(experiment_data, visualization_dir)
    visualize_belief_evolution(experiment_data, visualization_dir)
    
    return experiment_data


def visualize_trajectories(experiment_data: Dict, save_dir: str):
    """
    Visualize human and robot trajectories.
    
    Args:
        experiment_data: Experiment data dictionary
        save_dir: Directory to save visualization image
    """
    # Extract trajectory data
    human_states = torch.stack(experiment_data['human_states'])
    robot_states = torch.stack(experiment_data['robot_states'])
    
    # Extract positions
    human_positions = human_states[:, :2].numpy()
    robot_positions = robot_states[:, :2].numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw road environment
    # Horizontal road
    ax.add_patch(Rectangle(
        (-2.0, -0.1), 
        4.0, 0.2, 
        facecolor='gray',
        alpha=0.3
    ))
    # Vertical road
    ax.add_patch(Rectangle(
        (-0.1, -2.0), 
        0.2, 4.0, 
        facecolor='gray',
        alpha=0.3
    ))
    # Intersection
    ax.add_patch(Rectangle(
        (-0.1, -0.1), 
        0.2, 0.2, 
        facecolor='lightgray',
        alpha=0.3
    ))
    
    # Plot trajectories
    ax.plot(human_positions[:, 0], human_positions[:, 1], 'r-', linewidth=2, label='Human Trajectory')
    ax.plot(robot_positions[:, 0], robot_positions[:, 1], 'y-', linewidth=2, label='Robot Trajectory')
    
    # Mark start and end positions
    ax.plot(human_positions[0, 0], human_positions[0, 1], 'ro', markersize=10, label='Human Start')
    ax.plot(human_positions[-1, 0], human_positions[-1, 1], 'rx', markersize=10, label='Human End')
    ax.plot(robot_positions[0, 0], robot_positions[0, 1], 'yo', markersize=10, label='Robot Start')
    ax.plot(robot_positions[-1, 0], robot_positions[-1, 1], 'yx', markersize=10, label='Robot End')
    
    # Add timestep markers
    for i in range(0, len(human_positions), 2):
        ax.text(human_positions[i, 0], human_positions[i, 1], f"{i}", fontsize=8, color='darkred')
        ax.text(robot_positions[i, 0], robot_positions[i, 1], f"{i}", fontsize=8, color='darkgoldenrod')
    
    # Set plot limits and add labels
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Human and Robot Trajectories')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "trajectories.png"))
    plt.close(fig)


def visualize_reachable_sets(experiment_data: Dict, save_dir: str):
    """
    Visualize reachable sets at key time steps.
    
    Args:
        experiment_data: Experiment data dictionary
        save_dir: Directory to save visualization images
    """
    # Create directory for reachable set visualizations
    reachable_dir = os.path.join(save_dir, "reachable_sets")
    os.makedirs(reachable_dir, exist_ok=True)
    
    # Select time steps to visualize (every 3 steps)
    time_steps = list(range(0, len(experiment_data['reachable_sets']), 3))
    
    # If there are more than 5 steps, select first, three middle ones, and last
    if len(time_steps) > 5:
        middle_idx = len(time_steps) // 2
        time_steps = [0, 
                     time_steps[middle_idx-1], 
                     time_steps[middle_idx], 
                     time_steps[middle_idx+1], 
                     time_steps[-1]]
    
    # For each selected time step
    for t_idx, t in enumerate(time_steps):
        if t >= len(experiment_data['reachable_sets']):
            continue
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw road environment
        # Horizontal road
        ax.add_patch(Rectangle(
            (-2.0, -0.1), 
            4.0, 0.2, 
            facecolor='gray',
            alpha=0.3
        ))
        # Vertical road
        ax.add_patch(Rectangle(
            (-0.1, -2.0), 
            0.2, 4.0, 
            facecolor='gray',
            alpha=0.3
        ))
        # Intersection
        ax.add_patch(Rectangle(
            (-0.1, -0.1), 
            0.2, 0.2, 
            facecolor='lightgray',
            alpha=0.3
        ))
        
        # Get reachable sets for this time step
        reachable_sets = experiment_data['reachable_sets'][t]
        
        # Get human and robot states at this time step
        human_state = experiment_data['human_states'][t]
        robot_state = experiment_data['robot_states'][t]
        
        # Plot human and robot positions
        ax.plot(human_state[0].item(), human_state[1].item(), 'ro', markersize=10, label='Human')
        ax.plot(robot_state[0].item(), robot_state[1].item(), 'yo', markersize=10, label='Robot')
        
        # Plot reachable sets for next few time steps
        colors = ['lightblue', 'blue', 'darkblue', 'purple', 'red']
        
        for h in range(min(len(reachable_sets), len(colors))):
            reachable_set = reachable_sets[h]
            
            # Extract position coordinates
            positions = reachable_set[:, :2].detach().numpy()
            
            # Plot points
            ax.scatter(positions[:, 0], positions[:, 1], 
                      c=colors[h], alpha=0.3, s=20, 
                      label=f'Reachable t+{h}')
            
            # Try to compute and plot convex hull
            try:
                from scipy.spatial import ConvexHull
                if len(positions) >= 3:  # Need at least 3 points for a hull
                    hull = ConvexHull(positions)
                    
                    # Plot hull edges
                    for simplex in hull.simplices:
                        ax.plot(positions[simplex, 0], positions[simplex, 1], 
                               c=colors[h], alpha=0.7)
            except:
                # Skip if convex hull computation fails
                pass
        
        # Set plot limits and add labels
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Reachable Sets at Step {t}')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(reachable_dir, f"reachable_step_{t:03d}.png"))
        plt.close(fig)


def visualize_belief_evolution(experiment_data: Dict, save_dir: str):
    """
    Visualize belief evolution over time.
    
    Args:
        experiment_data: Experiment data dictionary
        save_dir: Directory to save visualization images
    """
    # Create directory for belief visualizations
    belief_dir = os.path.join(save_dir, "belief_evolution")
    os.makedirs(belief_dir, exist_ok=True)
    
    # 1. Create a grid of belief distributions at different time steps
    # Select time steps to visualize (max 6)
    steps = len(experiment_data['belief_states'])
    
    # Logic to select a representative set of time steps
    if steps <= 6:
        indices = list(range(steps))
    else:
        # First, middle, and last, plus three equally spaced
        step_size = steps // 5
        indices = [0]
        indices.extend([i * step_size for i in range(1, 5)])
        indices.append(steps - 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Get true internal state
    true_state = experiment_data['actual_internal_state']
    
    # Plot belief at each selected time step
    for i, idx in enumerate(indices):
        if i >= len(axes) or idx >= len(experiment_data['belief_states']):
            continue
            
        ax = axes[i]
        
        # Create temporary belief for plotting
        belief = IntervalBelief()
        belief.intervals = experiment_data['belief_states'][idx]['intervals']
        belief.probs = experiment_data['belief_states'][idx]['probs']
        
        # Plot belief
        belief.plot(ax=ax, title=f"Step {idx}", show_center=True)
        
        # Add true state
        ax.plot(true_state[0].item(), true_state[1].item(), 'g*', 
               markersize=12, label='True State')
        ax.legend(loc='upper right')
    
    # Hide any unused axes
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(belief_dir, "belief_grid.png"))
    plt.close(fig)
    
    # 2. Plot entropy and expected value evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot entropy over time
    ax1.plot(experiment_data['entropy_history'], 'b-', marker='o')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Belief Entropy')
    ax1.set_title('Entropy Reduction Over Time')
    ax1.grid(True)
    
    # Plot expected value convergence
    expected_values = torch.stack(experiment_data['expected_values'])
    
    ax2.plot(expected_values[:, 0].numpy(), 'b-', label='Expected Attentiveness')
    ax2.plot(expected_values[:, 1].numpy(), 'r-', label='Expected Driving Style')
    ax2.axhline(y=true_state[0].item(), linestyle='--', color='blue', 
               alpha=0.7, label='True Attentiveness')
    ax2.axhline(y=true_state[1].item(), linestyle='--', color='red', 
               alpha=0.7, label='True Driving Style')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value')
    ax2.set_title('Expected Internal State Over Time')
    ax2.legend()
    ax2.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(belief_dir, "belief_metrics.png"))
    plt.close(fig)


def main():
    """Main function to run experiments and generate visualizations."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define experiment configurations
    configs = [
        {"attentiveness": 0.8, "driving_style": 0.7, "name": "attentive_aggressive"},
        {"attentiveness": 0.3, "driving_style": 0.3, "name": "distracted_conservative"}
    ]
    
    # Run experiments
    for config in configs:
        print(f"\nRunning experiment: {config['name']}")
        config_dir = os.path.join(output_dir, config['name'])
        
        # Run experiment
        experiment_data = run_experiment(
            num_steps=15,
            attentiveness=config['attentiveness'],
            driving_style=config['driving_style'],
            visualization_dir=config_dir
        )
        
        print(f"Experiment {config['name']} completed.")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")