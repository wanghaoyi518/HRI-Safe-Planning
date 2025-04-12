# data_generation.py
import torch
import numpy as np
import pickle
import os
from dynamics import CarDynamics
from Agents.human import HumanAgent
from Agents.robot import RobotAgent
from environments import *
from rewards import *

class DataGenerator:
    """Generate synthetic data for training IRL models."""
    
    def __init__(self, dynamics):
        """Initialize with car dynamics model."""
        self.dynamics = dynamics
    def generate_dataset(self, num_episodes_per_config=5, output_dir="data"):
        """
        Generate dataset by running simulations with different human models.
        
        Args:
            num_episodes_per_config: Number of episodes to run per configuration
            output_dir: Directory to save data
            
        Returns:
            List of trajectories with ground truth internal states
        """
        os.makedirs(output_dir, exist_ok=True)
        dataset = []
        
        # Create grid of internal states to sample from
        attention_values = [0.2, 0.5, 0.8]  # Distracted to Attentive
        style_values = [0.2, 0.5, 0.8]      # Conservative to Aggressive
        
        # Run scenarios for different internal states
        for att in attention_values:
            for style in style_values:
                scenario_type = 'intersection'
                
                # Use the simple scenario creation function instead
                env, robot, human = create_intersection_scenario_simple()
                
                # Set human internal state
                human.internal_state = torch.tensor([att, style], dtype=torch.float32)
                
                # Set appropriate reward - pass the internal state as argument
                human.reward = create_parameterized_human_reward(human.internal_state)
                
                # Run multiple episodes
                for ep in range(num_episodes_per_config):
                    print(f"Running {scenario_type} scenario with att={att:.1f}, style={style:.1f}, ep={ep+1}/{num_episodes_per_config}")
                    trajectory = self._run_episode(env, robot, human)
                    
                    # Add metadata
                    trajectory['internal_state'] = human.internal_state.clone()
                    trajectory['scenario_type'] = scenario_type
                    trajectory['episode'] = ep
                    
                    dataset.append(trajectory)
                    
                    # Reset environment
                    env.reset()
        
        # Save the dataset
        filename = os.path.join(output_dir, "irl_dataset.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset saved to {filename} with {len(dataset)} trajectories")
        return dataset
    
    def _run_episode(self, env, robot, human, max_steps=20):
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
            if len(human.action_history) > 0:
                trajectory['human_actions'].append(human.action_history[-1].clone())
            if len(robot.action_history) > 0:
                trajectory['robot_actions'].append(robot.action_history[-1].clone())
            
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

def main():
    """Generate and save dataset for IRL training."""
    # Create dynamics model
    dynamics = CarDynamics(dt=0.1)
    
    # Create data generator
    generator = DataGenerator(dynamics)
    
    # Generate dataset
    dataset = generator.generate_dataset(num_episodes_per_config=1)
    
    print(f"Generated {len(dataset)} trajectory samples")

if __name__ == "__main__":
    main()