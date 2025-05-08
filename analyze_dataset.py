import pickle
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def analyze_irl_dataset(filepath="data/irl_dataset.pkl"):
    """
    Analyze and display information about the IRL dataset.
    
    Args:
        filepath: Path to the dataset pickle file
    """
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return
    
    # Load the dataset
    try:
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Successfully loaded dataset with {len(dataset)} trajectories.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Extract basic information
    internal_states = []
    scenario_types = []
    episode_lengths = []
    trajectory_steps = []
    
    # Create containers for statistical analysis
    att_values = []
    style_values = []
    
    # Analyze each trajectory
    for i, traj in enumerate(dataset):
        # Extract internal state
        if 'internal_state' in traj:
            internal_state = traj['internal_state']
            if isinstance(internal_state, torch.Tensor):
                att, style = internal_state.tolist()
                internal_states.append((round(att, 2), round(style, 2)))
                att_values.append(att)
                style_values.append(style)
        
        # Extract scenario type
        if 'scenario_type' in traj:
            scenario_types.append(traj['scenario_type'])
        
        # Extract episode length
        if 'episode_length' in traj:
            episode_lengths.append(traj['episode_length'])
        
        # Count trajectory steps (based on human states)
        if 'human_states' in traj:
            trajectory_steps.append(len(traj['human_states']))
        
        # Print first trajectory as an example (limited to first 5 entries)
        if i == 0:
            print("\nSample Trajectory Structure:")
            for key, value in traj.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"  {key}: {value[:5]} ... (plus {len(value)-5} more)")
                else:
                    print(f"  {key}: {value}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total trajectories: {len(dataset)}")
    
    # Analyze internal states
    if internal_states:
        unique_internal_states = set(internal_states)
        print(f"Unique internal states: {len(unique_internal_states)}")
        
        # Group by attentiveness and driving style
        att_ranges = defaultdict(int)
        style_ranges = defaultdict(int)
        
        for att, style in internal_states:
            att_ranges[att] += 1
            style_ranges[style] += 1
        
        print(f"Attentiveness values: {sorted(att_ranges.keys())}")
        print(f"Driving style values: {sorted(style_ranges.keys())}")
        
        # Calculate ranges
        print(f"Attentiveness range: [{min(att_values):.2f}, {max(att_values):.2f}]")
        print(f"Driving style range: [{min(style_values):.2f}, {max(style_values):.2f}]")
    
    # Analyze scenario types
    if scenario_types:
        scenario_counts = Counter(scenario_types)
        print(f"\nScenario types distribution:")
        for scenario, count in scenario_counts.most_common():
            print(f"  {scenario}: {count} trajectories")
    
    # Analyze episode and trajectory lengths
    if episode_lengths:
        print(f"\nEpisode length statistics:")
        print(f"  Average: {sum(episode_lengths)/len(episode_lengths):.2f}")
        print(f"  Min: {min(episode_lengths)}")
        print(f"  Max: {max(episode_lengths)}")
    
    if trajectory_steps:
        print(f"\nTrajectory steps statistics:")
        print(f"  Average: {sum(trajectory_steps)/len(trajectory_steps):.2f}")
        print(f"  Min: {min(trajectory_steps)}")
        print(f"  Max: {max(trajectory_steps)}")
    
    # Visualize internal state distribution if matplotlib is available
    try:
        if att_values and style_values:
            plt.figure(figsize=(10, 8))
            plt.scatter(att_values, style_values, alpha=0.6)
            plt.xlabel('Attentiveness')
            plt.ylabel('Driving Style')
            plt.title('Distribution of Internal States in Dataset')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.colorbar(plt.cm.ScalarMappable(), label='Density')
            plt.savefig('internal_state_distribution.png')
            print("\nVisualization of internal state distribution saved as 'internal_state_distribution.png'")
    except Exception as e:
        print(f"Could not create visualization: {e}")

if __name__ == "__main__":
    analyze_irl_dataset()