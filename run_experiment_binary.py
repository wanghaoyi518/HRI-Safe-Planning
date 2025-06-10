# run_experiment_binary.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import os
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

from dynamics import CarDynamics
from Agents.human import BinaryHumanAgent
from Agents.robot import RobotAgent
from reachability import BinaryReachabilityAnalysis
from environments import Intersection
from rewards import create_robot_reward, create_binary_human_reward


def create_binary_intersection_scenario(is_attentive=True):
    """
    Create an intersection scenario with robot and binary human agents.
    
    Args:
        is_attentive: True for attentive human, False for distracted
        
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
        planning_horizon=3,
        optimization_steps=10,
        info_gain_weight=2.0,
        safety_distance=0.2
    )
    
    # Enable binary belief mode
    robot.use_binary_belief(initial_p_attentive=0.5)  # Start with maximum uncertainty
    
    # Create binary human agent at west position, facing east
    human_init_state = torch.tensor([0.5, 0.0, np.pi, 0.0])
    
    human = BinaryHumanAgent(
        dynamics,
        human_init_state,
        is_attentive=is_attentive,
        name="human",
        color="red" if not is_attentive else "green",
        planning_horizon=4
    )
    
    # Register agents with environment
    env.register_agent(robot)
    env.register_agent(human)
    
    # Register human with robot
    robot.register_human(human)
    
    # Create binary reachability analyzer
    reachability_analyzer = BinaryReachabilityAnalysis(dynamics, human)
    robot.reachability_analyzer = reachability_analyzer
    
    return env, robot, human


def run_binary_experiment(
    num_steps: int = 20,
    true_is_attentive: bool = True,
    visualization_dir: str = "binary_visualizations"
):
    """
    Run the experiment with binary belief tracking.
    
    Args:
        num_steps: Number of simulation steps
        true_is_attentive: True human state (attentive or distracted)
        visualization_dir: Directory to save visualization images
        
    Returns:
        Dictionary with experiment data
    """
    human_type = "attentive" if true_is_attentive else "distracted"
    print(f"Running binary experiment with {human_type} human driver")
    
    # Create directory for visualizations
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Create scenario
    env, robot, human = create_binary_intersection_scenario(is_attentive=true_is_attentive)
    
    # Data collection
    experiment_data = {
        'human_states': [],
        'robot_states': [],
        'human_actions': [],
        'robot_actions': [],
        'reachable_sets': [],
        'binary_belief_history': [],
        'entropy_history': [],
        'true_is_attentive': true_is_attentive
    }
    
    # Simulation loop
    for t in range(num_steps):
        print(f"\n=== Simulation step {t+1}/{num_steps} ===")
        
        # Store current states
        experiment_data['human_states'].append(human.state.clone())
        experiment_data['robot_states'].append(robot.state.clone())
        
        # Store binary belief state
        belief_state = robot.get_binary_belief_state()
        experiment_data['binary_belief_history'].append(belief_state)
        
        # Compute and store entropy
        if hasattr(robot, 'binary_belief'):
            entropy = robot.binary_belief.entropy()
            experiment_data['entropy_history'].append(entropy.item())
            
            print(f"Binary belief: P(attentive)={belief_state['p_attentive']:.3f}, "
                  f"P(distracted)={belief_state['p_distracted']:.3f}")
            print(f"Entropy: {entropy.item():.4f}")
        
        # Create environment state
        env_state = {
            "human": human
        }
        
        # Robot computes control
        robot_action = robot.compute_control(t, robot.state, env_state)
        robot.set_action(robot_action)
        experiment_data['robot_actions'].append(robot_action.clone())
        
        # Human computes control
        human_action = human.compute_control(t, human.state, {"robot": robot})
        experiment_data['human_actions'].append(human_action.clone())
        
        # Execute environment step
        info = env.step()
        
        # Print current states
        print(f"Robot state: {robot.state}")
        print(f"Human state: {human.state}")
        print(f"Robot action: {robot_action}")
        print(f"Human action: {human_action}")
        
        # Update robot's binary belief based on observed human action
        print("\nUpdating binary belief based on observed human action...")
        robot.update_binary_belief("human", human_action)
        
        # Compute binary reachable sets
        if hasattr(robot, 'binary_belief'):
            p_dist, p_att = robot.binary_belief.get_probabilities()
            reachable_sets = robot.reachability_analyzer.compute_reachable_sets_binary(
                human.state,
                robot.state,
                p_attentive=p_att,
                p_distracted=p_dist,
                robot_controls=robot_action.unsqueeze(0).repeat(3, 1),
                time_horizon=3,
                samples_per_type=10
            )
            experiment_data['reachable_sets'].append(reachable_sets)
        
        # Check termination conditions
        if info.get('complete', False) or any(info.get('collisions', {}).values()):
            print(f"Simulation completed at step {t+1}")
            break
    
    # Create visualizations
    visualize_binary_trajectories(experiment_data, visualization_dir)
    visualize_binary_belief_evolution(experiment_data, visualization_dir)
    visualize_binary_reachable_sets(experiment_data, visualization_dir)
    
    return experiment_data


def visualize_binary_trajectories(experiment_data: Dict, save_dir: str):
    """
    Visualize trajectories with binary human type indication.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract trajectory data
    human_states = torch.stack(experiment_data['human_states'])
    robot_states = torch.stack(experiment_data['robot_states'])
    
    # Extract positions
    human_positions = human_states[:, :2].numpy()
    robot_positions = robot_states[:, :2].numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw intersection environment
    draw_intersection_environment(ax)
    
    # Color based on true human type
    human_color = 'green' if experiment_data['true_is_attentive'] else 'red'
    human_label = 'Human (Attentive)' if experiment_data['true_is_attentive'] else 'Human (Distracted)'
    
    # Plot trajectories
    ax.plot(human_positions[:, 0], human_positions[:, 1], 
            color=human_color, linewidth=2, label=human_label)
    ax.plot(robot_positions[:, 0], robot_positions[:, 1], 
            'y-', linewidth=2, label='Robot')
    
    # Mark start and end positions
    ax.plot(human_positions[0, 0], human_positions[0, 1], 'o', 
            color=human_color, markersize=10, label='Human Start')
    ax.plot(human_positions[-1, 0], human_positions[-1, 1], 'x', 
            color=human_color, markersize=10, label='Human End')
    ax.plot(robot_positions[0, 0], robot_positions[0, 1], 'yo', 
            markersize=10, label='Robot Start')
    ax.plot(robot_positions[-1, 0], robot_positions[-1, 1], 'yx', 
            markersize=10, label='Robot End')
    
    # Add timestep markers
    for i in range(0, len(human_positions), 3):
        ax.text(human_positions[i, 0], human_positions[i, 1], f"{i}", 
                fontsize=8, color='darkred')
        ax.text(robot_positions[i, 0], robot_positions[i, 1], f"{i}", 
                fontsize=8, color='darkgoldenrod')
    
    # Set plot limits and labels
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Binary Human-Robot Trajectories (True: {human_label})')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "binary_trajectories.png"))
    plt.close(fig)


def visualize_binary_belief_evolution(experiment_data: Dict, save_dir: str):
    """
    Visualize binary belief evolution over time.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract belief history
    belief_history = experiment_data['binary_belief_history']
    entropy_history = experiment_data['entropy_history']
    
    if not belief_history:
        print("No belief history to visualize")
        return
    
    # Extract probabilities over time
    p_attentive_history = [b['p_attentive'] for b in belief_history]
    p_distracted_history = [b['p_distracted'] for b in belief_history]
    time_steps = list(range(len(belief_history)))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Belief evolution
    ax1.plot(time_steps, p_attentive_history, 'g-', linewidth=2, label='P(Attentive)')
    ax1.plot(time_steps, p_distracted_history, 'r-', linewidth=2, label='P(Distracted)')
    
    # Add true state line
    true_prob = 1.0 if experiment_data['true_is_attentive'] else 0.0
    ax1.axhline(y=true_prob, color='black', linestyle='--', alpha=0.7,
                label=f'True State: {"Attentive" if experiment_data["true_is_attentive"] else "Distracted"}')
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Probability')
    ax1.set_title('Binary Belief Evolution')
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 2: Entropy evolution
    if entropy_history:
        ax2.plot(time_steps, entropy_history, 'b-', linewidth=2, marker='o')
        
        # Add maximum entropy line
        max_entropy = -2 * (0.5 * np.log(0.5))  # Binary entropy maximized at p=0.5
        ax2.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.5,
                    label=f'Max Entropy: {max_entropy:.3f}')
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Entropy')
        ax2.set_title('Belief Entropy Over Time')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "binary_belief_evolution.png"))
    plt.close(fig)


def visualize_binary_reachable_sets(experiment_data: Dict, save_dir: str):
    """
    Visualize reachable sets for binary human model.
    """
    reachable_dir = os.path.join(save_dir, "binary_reachable_sets")
    os.makedirs(reachable_dir, exist_ok=True)
    
    reachable_sets_history = experiment_data.get('reachable_sets', [])
    
    if not reachable_sets_history:
        print("No reachable sets to visualize")
        return
    
    # Select time steps to visualize
    time_steps = list(range(0, len(reachable_sets_history), 3))
    if len(time_steps) > 5:
        time_steps = time_steps[:5]
    
    for t in time_steps:
        if t >= len(reachable_sets_history):
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Draw environment
        draw_intersection_environment(ax)
        
        # Get states at this time
        human_state = experiment_data['human_states'][t]
        robot_state = experiment_data['robot_states'][t]
        
        # Plot current positions
        human_color = 'green' if experiment_data['true_is_attentive'] else 'red'
        ax.plot(human_state[0].item(), human_state[1].item(), 'o', 
                color=human_color, markersize=10, label='Human')
        ax.plot(robot_state[0].item(), robot_state[1].item(), 'yo', 
                markersize=10, label='Robot')
        
        # Get belief at this time
        belief = experiment_data['binary_belief_history'][t]
        p_att = belief['p_attentive']
        p_dist = belief['p_distracted']
        
        # Plot reachable sets
        reachable_sets = reachable_sets_history[t]
        
        # Use different colors for likely attentive vs distracted
        if p_att > 0.5:
            colors = ['lightgreen', 'green', 'darkgreen']
            dominant_type = "Likely Attentive"
        else:
            colors = ['lightcoral', 'red', 'darkred']
            dominant_type = "Likely Distracted"
        
        for h in range(min(len(reachable_sets), len(colors))):
            reachable_set = reachable_sets[h]
            
            # Extract positions
            positions = reachable_set[:, :2].detach().numpy()
            
            # Plot points
            ax.scatter(positions[:, 0], positions[:, 1], 
                      c=colors[h], alpha=0.5, s=30, 
                      label=f'Reachable t+{h}')
            
            # Try to compute convex hull
            try:
                from scipy.spatial import ConvexHull
                if len(positions) >= 3:
                    hull = ConvexHull(positions)
                    
                    # Plot hull
                    for simplex in hull.simplices:
                        ax.plot(positions[simplex, 0], positions[simplex, 1], 
                               c=colors[h], alpha=0.7)
            except:
                pass
        
        # Set plot properties
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Binary Reachable Sets at Step {t}\n'
                    f'P(Attentive)={p_att:.2f}, P(Distracted)={p_dist:.2f} ({dominant_type})')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(reachable_dir, f"binary_reachable_step_{t:03d}.png"))
        plt.close(fig)


def draw_intersection_environment(ax):
    """
    Helper function to draw intersection environment.
    """
    road_width = 0.2
    intersection_size = 0.3
    half_size = intersection_size / 2
    
    # Draw horizontal road
    h_road = Rectangle(
        (-2.0, -road_width/2), 
        4.0, road_width,
        facecolor='gray',
        alpha=0.3
    )
    ax.add_patch(h_road)
    
    # Draw vertical road
    v_road = Rectangle(
        (-road_width/2, -2.0), 
        road_width, 4.0,
        facecolor='gray',
        alpha=0.3
    )
    ax.add_patch(v_road)
    
    # Draw intersection
    intersection_box = Rectangle(
        (-half_size, -half_size), 
        intersection_size, intersection_size,
        facecolor='lightgray',
        alpha=0.3
    )
    ax.add_patch(intersection_box)
    
    # Draw boundary
    boundary_box = Rectangle(
        (-2.0, -2.0),
        4.0, 4.0,
        fill=False,
        edgecolor='red',
        linestyle=':',
        linewidth=1.5
    )
    ax.add_patch(boundary_box)
    
    # Add human goal
    human_goal = (-2.0, 0.0)
    ax.plot(human_goal[0], human_goal[1], 'r*', markersize=15, label='Human Goal')


def main():
    """Main function to run binary experiments."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create timestamp for this experiment run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create output directory with timestamp
    output_dir = os.path.join("binary_experiment_results", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created experiment directory: {output_dir}")
    
    # Run experiments for both human types
    configs = [
        {"true_is_attentive": True, "name": "attentive_human"},
        {"true_is_attentive": False, "name": "distracted_human"}
    ]
    
    # Run experiments
    all_results = {}
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Running experiment: {config['name']}")
        print(f"{'='*50}")
        
        config_dir = os.path.join(output_dir, config['name'])
        
        # Run experiment
        experiment_data = run_binary_experiment(
            num_steps=20,
            true_is_attentive=config['true_is_attentive'],
            visualization_dir=config_dir
        )
        
        # Store results
        all_results[config['name']] = experiment_data
        
        # Print final belief
        final_belief = experiment_data['binary_belief_history'][-1]
        print(f"\nFinal belief for {config['name']}:")
        print(f"  P(attentive) = {final_belief['p_attentive']:.3f}")
        print(f"  P(distracted) = {final_belief['p_distracted']:.3f}")
        print(f"  Most likely: {final_belief['most_likely']}")
        
        correct = (final_belief['most_likely'] == 'attentive') == config['true_is_attentive']
        print(f"  Correct classification: {correct}")
    
    print(f"\nAll experiments completed. Results saved to {output_dir}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")