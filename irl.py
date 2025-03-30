# irl.py
import torch
import numpy as np
from typing import List, Dict, Callable
import matplotlib.pyplot as plt

class MaxEntIRL:
    """Maximum Entropy IRL implementation for learning human reward functions."""
    
    def __init__(self, feature_functions: List[Callable]):
        """
        Initialize IRL with feature functions.
        
        Args:
            feature_functions: List of feature functions that take (time, state, action)
        """
        self.feature_functions = feature_functions
        self.feature_names = [f.__name__ for f in feature_functions]
        self.weights = None
        
    def extract_features(self, trajectory):
        """Extract features from a trajectory."""
        features = {name: 0.0 for name in self.feature_names}
        
        states = trajectory['human_states']
        actions = trajectory['human_actions']
        
        # Sum feature values across trajectory
        for t in range(min(len(states)-1, len(actions))):
            state = states[t]
            action = actions[t]
            
            for i, feature_fn in enumerate(self.feature_functions):
                name = self.feature_names[i]
                value = feature_fn(t, state, action)
                features[name] += value.item()
        
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
        """
        Train IRL model using Maximum Entropy method.
        
        Args:
            demonstrations: List of demonstration trajectories
            learning_rate: Learning rate for optimization
            num_iterations: Number of optimization iterations
            regularization: L2 regularization strength
            
        Returns:
            Learned reward weights
        """
        # Initialize weights
        self.weights = torch.ones(len(self.feature_functions), requires_grad=True)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam([self.weights], lr=learning_rate)
        
        # Compute feature expectations from demonstrations
        demo_expectations = self.compute_feature_expectations(demonstrations)
        
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
                
                # Compute gradients of reward w.r.t. actions
                grads = []
                for t, reward in enumerate(rewards):
                    action = actions[t]
                    if action.requires_grad:
                        action.retain_grad()
                        reward.backward(retain_graph=True)
                        if action.grad is not None:
                            grads.append(action.grad.clone())
                        action.grad = None
                
                # Use approximation for Hessian inverse and determinant
                # This is a simplified version of the Laplace approximation
                if grads:
                    grad_vector = torch.cat([g.flatten() for g in grads])
                    hessian_approx = -torch.eye(len(grad_vector)) * regularization
                    
                    # Maximum entropy objective (simplified)
                    maxent_obj = torch.sum(grad_vector**2) / (2 * regularization)
                    
                    # Add to batch loss (negative because we're maximizing)
                    batch_loss -= (total_reward - maxent_obj)
                else:
                    # Fallback to direct reward optimization
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
        
        # Return learned weights as dictionary
        return {name: weight.item() for name, weight in zip(self.feature_names, self.weights)}

class StateParameterizedIRL:
    """Learn reward functions parameterized by internal states."""
    
    def __init__(self, feature_functions):
        """Initialize with feature functions."""
        self.feature_functions = feature_functions
        self.internal_state_models = {}
    
    def train(self, dataset, num_bins=2):
        """
        Train models for different internal states.
        
        Args:
            dataset: List of trajectories with ground truth internal states
            num_bins: Number of bins for discretizing internal state space
            
        Returns:
            Dictionary mapping internal state bins to learned weights
        """
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
    
    def create_reward_function(self, internal_state):
        """Create reward function for a given internal state."""
        att, style = internal_state.tolist() if isinstance(internal_state, torch.Tensor) else internal_state
        
        # Find the bin that contains this internal state
        matching_bins = []
        for bin_key, model in self.internal_state_models.items():
            att_range = model['att_range']
            style_range = model['style_range']
            
            if att_range[0] <= att <= att_range[1] and style_range[0] <= style <= style_range[1]:
                matching_bins.append(bin_key)
        
        if not matching_bins:
            # Find closest bin if no exact match
            closest_bin = None
            min_distance = float('inf')
            
            for bin_key, model in self.internal_state_models.items():
                att_center = sum(model['att_range']) / 2
                style_center = sum(model['style_range']) / 2
                
                distance = ((att - att_center)**2 + (style - style_center)**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_bin = bin_key
                    
            matching_bins = [closest_bin] if closest_bin is not None else []
        
        if not matching_bins:
            raise ValueError("No models available for this internal state")
            
        # Use the first matching bin (or closest bin)
        bin_key = matching_bins[0]
        model = self.internal_state_models[bin_key]
        weights = model['weights']
        
        # Create reward function
        def reward_function(t, state, action):
            reward = 0.0
            for i, feature_fn in enumerate(self.feature_functions):
                name = feature_fn.__name__
                if name in weights:
                    feature_value = feature_fn(t, state, action)
                    reward += weights[name] * feature_value
            return reward
        
        return reward_function
    
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
        return fig