# data_generation.py
import torch
import numpy as np
from environments import create_highway_scenario, create_intersection_scenario

class DataGenerator:
    """Generate synthetic data for training IRL models."""
    
    def __init__(self, dynamics):
        """Initialize with car dynamics model."""
        self.dynamics = dynamics
        
    def generate_dataset(self, num_scenarios=10, episodes_per_scenario=5):
        """
        Generate dataset by running simulations with different human models.
        
        Returns:
            List of trajectories with ground truth internal states
        """
        dataset = []
        
        # Create grid of internal states to sample from
        attention_values = [0.2, 0.5, 0.8]  # Distracted to Attentive
        style_values = [0.2, 0.5, 0.8]      # Conservative to Aggressive
        
        # Run scenarios for different internal states
        for att in attention_values:
            for style in style_values:
                # Run both highway and intersection scenarios
                for scenario_type in ['highway', 'intersection']:
                    # Create environment and agents
                    if scenario_type == 'highway':
                        env, robot, human = create_highway_scenario()
                    else:
                        env, robot, human = create_intersection_scenario()
                    
                    # Set human internal state
                    human.internal_state = torch.tensor([att, style], dtype=torch.float32)
                    
                    # Run multiple episodes
                    for _ in range(episodes_per_scenario):
                        trajectory = self._run_episode(env, robot, human)
                        
                        # Add metadata
                        trajectory['internal_state'] = human.internal_state.clone()
                        trajectory['scenario_type'] = scenario_type
                        
                        dataset.append(trajectory)
                        env.reset()
        
        return dataset
    
    def _run_episode(self, env, robot, human, max_steps=50):
        """Run a single episode and collect trajectory data."""
        trajectory = {
            'human_states': [],
            'human_actions': [],
            'robot_states': [],
            'robot_actions': []
        }
        
        # Run simulation
        for step in range(max_steps):
            # Record current states
            trajectory['human_states'].append(human.state.clone())
            trajectory['robot_states'].append(robot.state.clone())
            
            # Record actions (if available)
            if hasattr(human, 'action') and human.action is not None:
                trajectory['human_actions'].append(human.action.clone())
            if hasattr(robot, 'action') and robot.action is not None:
                trajectory['robot_actions'].append(robot.action.clone())
            
            # Step environment
            info = env.step()
            
            # Stop if episode complete
            if info.get('complete', False) or any(info.get('collisions', {}).values()):
                break
        
        # Add final states
        trajectory['human_states'].append(human.state.clone())
        trajectory['robot_states'].append(robot.state.clone())
        
        # Add episode metadata
        trajectory['episode_length'] = step + 1
        
        return trajectory
    
    def save_dataset(self, dataset, filename='irl_dataset.pkl'):
        """Save dataset to file."""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)
    
    def load_dataset(self, filename='irl_dataset.pkl'):
        """Load dataset from file."""
        import pickle
        with open(filename, 'rb') as f:
            return pickle.load(f)