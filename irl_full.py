import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
import pickle
import os
import matplotlib.pyplot as plt

# Import our custom modules as needed
from dynamics import CarDynamics
from features import Feature, speed_feature, lane_keeping_feature, collision_avoidance_feature
from environments import create_highway_scenario, create_intersection_scenario


class IRL:
    """
    Inverse Reinforcement Learning implementation for human driver modeling.
    Combines data collection, feature extraction, and reward learning in one class.
    """
    
    def __init__(self, dynamics, feature_functions=None):
        """
        Initialize IRL framework.
        
        Args:
            dynamics: Vehicle dynamics model
            feature_functions: List of feature functions (optional)
        """
        self.dynamics = dynamics
        self.feature_functions = feature_functions or []
        self.dataset = []
        self.internal_state_map = {}
        
    def add_feature(self, feature_fn):
        """Add a feature function to the IRL model."""
        self.feature_functions.append(feature_fn)
        
    def generate_data(self, scenarios=None, output_dir=None):
        """
        Generate synthetic data by simulating interactions with 
        parameterized human models.
        
        Args:
            scenarios: List of scenario configurations
            output_dir: Directory to save generated data
        
        Returns:
            List of collected trajectories
        """
        # Default scenarios if none provided
        if scenarios is None:
            scenarios = self._default_scenarios()
            
        dataset = []
        
        # Run each scenario
        for scenario in scenarios:
            # Create environment and agents
            if scenario['type'] == 'highway':
                env, robot, human = create_highway_scenario(
                    robot_lane=scenario.get('robot_lane', 1),
                    human_lane=scenario.get('human_lane', 0)
                )
            else:  # intersection
                env, robot, human = create_intersection_scenario(
                    robot_direction=scenario.get('robot_direction', 'south'),
                    human_direction=scenario.get('human_direction', 'west')
                )
                
            # Set human internal state
            human.internal_state = torch.tensor(
                scenario['internal_state'], 
                dtype=torch.float32
            )
            
            # Run episodes
            for ep in range(scenario.get('episodes', 5)):
                # Run episode
                env.reset()
                episode_data = self._run_episode(env, robot, human)
                
                # Add metadata
                episode_data['internal_state'] = human.internal_state.clone()
                episode_data['scenario_type'] = scenario['type']
                
                dataset.append(episode_data)
                
        # Save dataset if output directory provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'irl_dataset.pkl'), 'wb') as f:
                pickle.dump(dataset, f)
                
        self.dataset = dataset
        return dataset
                
    def _run_episode(self, env, robot, human, max_steps=50):
        """Run a single episode and collect trajectory data."""
        states_h = []
        actions_h = []
        states_r = []
        actions_r = []
        
        # Run simulation
        for step in range(max_steps):
            # Save current states and actions
            states_h.append(human.state.clone())
            if len(human.action_history) > 0:
                actions_h.append(human.action_history[-1].clone())
                
            states_r.append(robot.state.clone())
            if len(robot.action_history) > 0:
                actions_r.append(robot.action_history[-1].clone())
            
            # Step environment
            info = env.step()
            
            # Check if episode is complete
            if info['complete'] or any(info['collisions'].values()):
                break
                
        # Add final states
        states_h.append(human.state.clone())
        states_r.append(robot.state.clone())
        
        return {
            'human_states': states_h,
            'human_actions': actions_h,
            'robot_states': states_r,
            'robot_actions': actions_r,
            'episode_length': step + 1
        }
    
    def _default_scenarios(self):
        """Create default scenarios covering different internal states."""
        scenarios = []
        
        # Create grid of internal states to sample from
        attention_values = [0.2, 0.5, 0.8]  # Distracted to Attentive
        style_values = [0.2, 0.5, 0.8]      # Conservative to Aggressive
        
        # Create scenarios for each combination
        for att in attention_values:
            for style in style_values:
                # Highway scenario
                scenarios.append({
                    'type': 'highway',
                    'internal_state': [att, style],
                    'episodes': 5
                })
                
                # Intersection scenario
                scenarios.append({
                    'type': 'intersection',
                    'internal_state': [att, style],
                    'episodes': 5
                })
                
        return scenarios
    
    def extract_features(self, trajectory):
        """Extract features for a single trajectory."""
        # Get state and action sequences
        states = trajectory['human_states']
        actions = trajectory['human_actions']
        
        # Initialize feature values
        feature_values = {f.__name__: 0.0 for f in self.feature_functions}
        
        # Compute features for each timestep
        for t in range(len(actions)):
            state = states[t]
            action = actions[t]
            
            # Compute each feature
            for feature_fn in self.feature_functions:
                # Accumulate feature value
                value = feature_fn(t, state, action)
                feature_values[feature_fn.__name__] += value.item()
                
        # Normalize by trajectory length
        for name in feature_values:
            if len(actions) > 0:
                feature_values[name] /= len(actions)
            
        return feature_values
    
    def compute_feature_expectations(self, trajectories):
        """Compute empirical feature expectations from trajectories."""
        # Initialize expectations
        expectations = {f.__name__: 0.0 for f in self.feature_functions}
        count = 0
        
        # Compute average feature values across trajectories
        for traj in trajectories:
            traj_features = self.extract_features(traj)
            for name, value in traj_features.items():
                expectations[name] += value
            count += 1
                
        # Normalize by number of trajectories
        if count > 0:
            for name in expectations:
                expectations[name] /= count
            
        return expectations
    
    def train(self, dataset=None, internal_state_bins=2, num_iterations=100):
        """
        Train IRL models for different internal states.
        
        Args:
            dataset: List of demonstrations (uses self.dataset if None)
            internal_state_bins: Number of bins for discretizing internal states
            num_iterations: Number of optimization iterations
            
        Returns:
            Mapping from internal state to reward weights
        """
        if dataset is None:
            dataset = self.dataset
            
        if not dataset:
            raise ValueError("No dataset available. Generate data first.")
            
        # Bin internal states
        att_bins = np.linspace(0, 1, internal_state_bins+1)
        style_bins = np.linspace(0, 1, internal_state_bins+1)
        
        # Group demonstrations by internal state bin
        grouped_demos = {}
        
        for demo in dataset:
            internal_state = demo['internal_state']
            att, style = internal_state
            
            # Find bin indices
            att_idx = np.digitize(att.item(), att_bins) - 1
            style_idx = np.digitize(style.item(), style_bins) - 1
            
            # Clip to valid range
            att_idx = max(0, min(att_idx, internal_state_bins-1))
            style_idx = max(0, min(style_idx, internal_state_bins-1))
            
            # Create bin key
            bin_key = (att_idx, style_idx)
            
            if bin_key not in grouped_demos:
                grouped_demos[bin_key] = []
                
            grouped_demos[bin_key].append(demo)
            
        # Train for each bin
        for bin_key, demos in grouped_demos.items():
            # Skip bins with too few demonstrations
            if len(demos) < 3:
                print(f"Skipping bin {bin_key} with only {len(demos)} demos")
                continue
                
            # Compute feature expectations
            feature_expectations = self.compute_feature_expectations(demos)
            
            # Initialize weights
            weights = {name: 1.0 for name in feature_expectations.keys()}
            
            # Optimize weights
            weights = self._optimize_weights(
                demos, 
                feature_expectations, 
                weights,
                num_iterations
            )
            
            # Store mapping from bin to weights
            att_idx, style_idx = bin_key
            att_center = (att_bins[att_idx] + att_bins[att_idx+1]) / 2
            style_center = (style_bins[style_idx] + style_bins[style_idx+1]) / 2
            
            self.internal_state_map[bin_key] = {
                'internal_state': [att_center, style_center],
                'weights': weights
            }
            
        return self.internal_state_map
    
    def _optimize_weights(self, demos, feature_expectations, init_weights, num_iterations):
        """
        Optimize reward weights using maximum entropy IRL.
        
        Simplified version of the optimization in the original paper.
        
        Args:
            demos: List of demonstrations
            feature_expectations: Target feature expectations
            init_weights: Initial weights
            num_iterations: Number of optimization iterations
            
        Returns:
            Optimized weights
        """
        # Initialize weights
        weights = {name: init_weights[name] for name in init_weights}
        weights_tensor = torch.tensor(
            [weights[name] for name in sorted(weights.keys())],
            requires_grad=True
        )
        
        # Initialize optimizer
        optimizer = torch.optim.Adam([weights_tensor], lr=0.01)
        
        # Names in sorted order
        names = sorted(weights.keys())
        
        # Training loop
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Compute loss for each demonstration
            loss = 0.0
            for demo in demos:
                # Extract trajectory data
                states = demo['human_states']
                actions = demo['human_actions']
                
                # Skip if no actions
                if len(actions) == 0:
                    continue
                
                # Compute trajectory features
                traj_features = self.extract_features(demo)
                
                # Compute reward as weighted sum of features
                reward = 0.0
                for i, name in enumerate(names):
                    reward += weights_tensor[i] * traj_features[name]
                
                # Add reward to loss (negative because we want to maximize)
                loss -= reward
                
            # Simple L2 regularization
            reg_term = 0.1 * torch.sum(weights_tensor**2)
            loss += reg_term
            
            # Update weights
            loss.backward()
            optimizer.step()
            
            # Print progress
            if iteration % 20 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}")
                print(f"Weights: {weights_tensor.detach().numpy()}")
                
        # Update weights dictionary
        for i, name in enumerate(names):
            weights[name] = weights_tensor[i].item()
            
        return weights
    
    def create_reward_function(self, internal_state):
        """Create reward function for a given internal state."""
        # Find closest bin
        closest_bin = None
        min_distance = float('inf')
        
        for bin_key, data in self.internal_state_map.items():
            bin_state = data['internal_state']
            distance = np.linalg.norm(
                np.array(internal_state) - np.array(bin_state)
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_bin = bin_key
                
        if closest_bin is None:
            raise ValueError("No trained reward model available")
            
        # Get weights for closest bin
        weights = self.internal_state_map[closest_bin]['weights']
        
        # Create reward function
        def reward_fn(t, state, action):
            total_reward = 0.0
            for i, feature_fn in enumerate(self.feature_functions):
                feature_name = feature_fn.__name__
                if feature_name in weights:
                    feature_value = feature_fn(t, state, action)
                    total_reward += weights[feature_name] * feature_value
            return total_reward
            
        return reward_fn
    
    def save_model(self, filename):
        """Save trained model to file."""
        model_data = {
            'internal_state_map': self.internal_state_map,
            'feature_names': [f.__name__ for f in self.feature_functions]
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, filename):
        """Load trained model from file."""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
            
        self.internal_state_map = model_data['internal_state_map']
        
    def visualize_weights(self):
        """Visualize learned weights for different internal states."""
        if not self.internal_state_map:
            raise ValueError("No trained model available")
            
        # Get all feature names
        feature_names = set()
        for data in self.internal_state_map.values():
            feature_names.update(data['weights'].keys())
            
        feature_names = sorted(feature_names)
        
        # Get all internal states
        internal_states = []
        for data in self.internal_state_map.values():
            internal_states.append(data['internal_state'])
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot weights for each internal state
        for i, bin_key in enumerate(self.internal_state_map):
            data = self.internal_state_map[bin_key]
            weights = data['weights']
            internal_state = data['internal_state']
            
            # Get weights in consistent order
            weight_values = [weights.get(name, 0.0) for name in feature_names]
            
            # Plot bar group
            x = np.arange(len(feature_names))
            width = 0.8 / len(self.internal_state_map)
            offset = i * width - 0.4 + width/2
            
            bars = ax.bar(x + offset, weight_values, width, 
                         label=f"Att={internal_state[0]:.1f}, Style={internal_state[1]:.1f}")
            
        # Set labels
        ax.set_xlabel('Features')
        ax.set_ylabel('Weights')
        ax.set_title('Learned Reward Weights for Different Internal States')
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig


def create_default_features():
    """Create a set of default feature functions for driving."""
    features = []
    
    # Add speed feature
    features.append(speed_feature(target_speed=1.0))
    
    # Add lane keeping feature
    features.append(lane_keeping_feature([0, 1, 0], 0.13))
    
    # Add collision avoidance feature (simplified)
    def collision_feature(t, state, action):
        # Simple distance-based penalty
        # In real implementation, this would use information about other agents
        return torch.tensor(-1.0 * (action[0]**2 + action[1]**2), dtype=torch.float32)
    
    features.append(collision_feature)
    
    return features


# Example usage
if __name__ == "__main__":
    # Create dynamics model
    dynamics = CarDynamics(dt=0.1)
    
    # Create IRL framework
    irl = IRL(dynamics, feature_functions=create_default_features())
    
    # Generate data
    print("Generating dataset...")
    dataset = irl.generate_data(output_dir="data")
    
    # Train IRL models
    print("Training IRL models...")
    internal_state_map = irl.train(num_iterations=50)
    
    # Save model
    irl.save_model("irl_model.pkl")
    
    # Visualize weights
    irl.visualize_weights()
    
    # Create reward function for specific internal state
    internal_state = [0.7, 0.3]  # High attentiveness, moderate conservativeness
    reward_fn = irl.create_reward_function(internal_state)
    print(f"Created reward function for internal state {internal_state}")