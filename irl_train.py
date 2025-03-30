# irl_train.py
import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from dynamics import CarDynamics
from features import (
    speed_preference_feature, 
    lane_following_feature,
    obstacle_avoidance_feature, 
    control_smoothness_feature
)

class MaxEntIRL:
    """Maximum Entropy IRL implementation for learning human reward functions."""
    
    def __init__(self, feature_functions):
        """Initialize IRL with feature functions."""
        self.feature_functions = feature_functions
        self.feature_names = [f.__name__ if hasattr(f, '__name__') else f.__class__.__name__ 
                             for f in feature_functions]
        self.weights = None
        
    def extract_features(self, trajectory):
        """Extract features from a trajectory."""
        features = {name: 0.0 for name in self.feature_names}
        
        states = trajectory['human_states']
        actions = trajectory['human_actions']
        
        # Skip if no actions
        if len(actions) == 0:
            return features
            
        # Sum feature values across trajectory
        for t in range(min(len(states)-1, len(actions))):
            state = states[t]
            action = actions[t]
            
            for i, feature_fn in enumerate(self.feature_functions):
                name = self.feature_names[i]
                value = feature_fn(t, state, action)
                features[name] += value.item() if isinstance(value, torch.Tensor) else value
        
        # Normalize by trajectory length
        traj_length = min(len(states)-1, len(actions))
        if traj_length > 0:
            for name in features:
                features[name] /= traj_length
                
        return features
    
    def compute_feature_expectations(self, demonstrations):
        """Compute empirical feature expectations from demonstrations."""
        # Initialize expectation accumulators
        expectations = {name: 0.0 for name in self.feature_names}
        
        # Sum over all demonstrations
        for demo in demonstrations:
            features = self.extract_features(demo)
            for name, value in features.items():
                expectations[name] += value
        
        # Normalize by number of demonstrations
        for name in expectations:
            expectations[name] /= len(demonstrations)
            
        return expectations
    
    def train(self, demonstrations, learning_rate=0.01, num_iterations=100, regularization=0.1):
        """Train IRL model using Maximum Entropy method."""
        # Initialize weights
        self.weights = torch.ones(len(self.feature_functions), requires_grad=True)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam([self.weights], lr=learning_rate)
        
        # Training loop
        losses = []
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Compute loss over all demonstrations
            batch_loss = 0.0
            for demo in demonstrations:
                # Extract trajectory data
                states = demo['human_states']
                actions = demo['human_actions']
                
                # Skip empty trajectories
                if len(actions) == 0:
                    continue
                
                # Compute reward for each timestep
                rewards = []
                for t in range(min(len(states)-1, len(actions))):
                    state = states[t]
                    action = actions[t]
                    
                    # Compute weighted sum of features
                    reward = 0.0
                    for i, feature_fn in enumerate(self.feature_functions):
                        feature_value = feature_fn(t, state, action)
                        reward += self.weights[i] * feature_value
                        
                    rewards.append(reward)
                
                if len(rewards) == 0:
                    continue
                    
                total_reward = torch.stack(rewards).sum()
                
                # Add direct reward maximization objective
                batch_loss -= total_reward
            
            # Apply regularization to weights
            l2_reg = regularization * torch.sum(self.weights**2)
            batch_loss += l2_reg
            
            # Update weights
            batch_loss.backward()
            optimizer.step()
            
            # Track loss
            losses.append(batch_loss.item())
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {batch_loss.item()}")
                print(f"Weights: {self.weights.detach().numpy()}")
        
        # Plot training curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('IRL Training Loss')
        plt.savefig('irl_training_loss.png')
        
        # Return learned weights as dictionary
        return {name: weight.item() for name, weight in zip(self.feature_names, self.weights)}

class StateParameterizedIRL:
    """Learn reward functions parameterized by internal states."""
    
    def __init__(self, feature_functions):
        """Initialize with feature functions."""
        self.feature_functions = feature_functions
        self.internal_state_models = {}
    
    def train(self, dataset, num_bins=2):
        """Train models for different internal states."""
        # Create bins for attentiveness and driving style
        att_bins = np.linspace(0, 1, num_bins+1)
        style_bins = np.linspace(0, 1, num_bins+1)
        
        # Group demonstrations by internal state bin
        binned_data = {}
        
        for demo in dataset:
            # Extract internal state
            internal_state = demo['internal_state']
            att, style = internal_state.tolist()
            
            # Find bin indices
            att_idx = np.digitize(att, att_bins) - 1
            style_idx = np.digitize(style, style_bins) - 1
            
            # Clip to valid range
            att_idx = max(0, min(att_idx, num_bins-1))
            style_idx = max(0, min(style_idx, num_bins-1))
            
            # Create bin key
            bin_key = (att_idx, style_idx)
            
            if bin_key not in binned_data:
                binned_data[bin_key] = []
            
            binned_data[bin_key].append(demo)
        
        # Train model for each bin
        for bin_key, demos in binned_data.items():
            if len(demos) < 3:  # Skip bins with too few samples
                print(f"Skipping bin {bin_key} with only {len(demos)} demonstrations")
                continue
                
            print(f"Training model for bin {bin_key} with {len(demos)} demonstrations")
            
            # Create and train IRL model
            irl = MaxEntIRL(self.feature_functions)
            weights = irl.train(demos)
            
            # Store model
            att_idx, style_idx = bin_key
            att_range = (att_bins[att_idx], att_bins[att_idx+1])
            style_range = (style_bins[style_idx], style_bins[style_idx+1])
            
            self.internal_state_models[bin_key] = {
                'att_range': att_range,
                'style_range': style_range,
                'weights': weights
            }
        
        return self.internal_state_models
    
    def plot_weight_landscape(self):
        """Visualize how weights vary with internal state."""
        if not self.internal_state_models:
            print("No models trained yet")
            return
            
        # Get all weight names
        all_weight_names = set()
        for model in self.internal_state_models.values():
            all_weight_names.update(model['weights'].keys())
            
        # Create plot grid
        n_weights = len(all_weight_names)
        fig, axes = plt.subplots(1, n_weights, figsize=(n_weights*4, 4))
        if n_weights == 1:
            axes = [axes]
            
        for i, weight_name in enumerate(sorted(all_weight_names)):
            ax = axes[i]
            
            # Collect weight values and positions
            positions = []
            values = []
            
            for bin_key, model in self.internal_state_models.items():
                att_center = sum(model['att_range']) / 2
                style_center = sum(model['style_range']) / 2
                
                weight = model['weights'].get(weight_name, 0.0)
                
                positions.append((att_center, style_center))
                values.append(weight)
                
            # Convert to arrays
            positions = np.array(positions)
            
            # Create scatter plot
            scatter = ax.scatter(
                positions[:, 0], positions[:, 1], 
                c=values, cmap='viridis', 
                s=100, alpha=0.8
            )
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax)
            
            # Add labels
            ax.set_xlabel('Attentiveness')
            ax.set_ylabel('Driving Style')
            ax.set_title(f'Weight: {weight_name}')
            
        plt.tight_layout()
        plt.savefig('weight_landscape.png')
        return fig
    
    def save_models(self, filename='irl_models.pkl'):
        """Save trained models to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.internal_state_models, f)
            
    def load_models(self, filename='irl_models.pkl'):
        """Load trained models from file."""
        with open(filename, 'rb') as f:
            self.internal_state_models = pickle.load(f)

def main():
    """Load data and train IRL models."""
    # Load dataset
    data_file = "data/irl_dataset.pkl"
    
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Please run data_generation.py first.")
        return
        
    with open(data_file, 'rb') as f:
        dataset = pickle.load(f)
        
    print(f"Loaded dataset with {len(dataset)} trajectories")
    
    # Create feature functions
    features = [
        speed_preference_feature(1.0),
        lane_following_feature(0.0),
        obstacle_avoidance_feature(),
        control_smoothness_feature()
    ]
    
    # Create state-parameterized IRL model
    irl = StateParameterizedIRL(features)
    
    # Train models
    models = irl.train(dataset, num_bins=3)
    
    # Display learned models
    print("\nLearned models:")
    for bin_key, model in models.items():
        print(f"Bin {bin_key}:")
        print(f"  Attentiveness range: {model['att_range']}")
        print(f"  Driving style range: {model['style_range']}")
        print(f"  Weights: {model['weights']}")
    
    # Visualize weight landscape
    irl.plot_weight_landscape()
    
    # Save models
    irl.save_models('irl_models.pkl')
    
    print("IRL training complete. Models saved to irl_models.pkl")

if __name__ == "__main__":
    main()