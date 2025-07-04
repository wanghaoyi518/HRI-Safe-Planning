import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Interval:
    """
    Represents a rectangular interval in the internal state space.
    """
    
    def __init__(self, 
                 att_range: Tuple[float, float],
                 style_range: Tuple[float, float]):
        """
        Initialize interval with attentiveness and driving style ranges.
        
        Args:
            att_range: (min, max) for attentiveness
            style_range: (min, max) for driving style
        """
        self.att_range = att_range
        self.style_range = style_range
        
    def __str__(self):
        return f"Interval(att=[{self.att_range[0]:.2f}, {self.att_range[1]:.2f}], " \
               f"style=[{self.style_range[0]:.2f}, {self.style_range[1]:.2f}])"
               
    def contains(self, phi: torch.Tensor) -> bool:
        """
        Check if internal state is within this interval.
        
        Args:
            phi: Internal state [attentiveness, driving_style]
            
        Returns:
            True if phi is in the interval
        """
        att, style = phi
        return (self.att_range[0] <= att <= self.att_range[1] and
                self.style_range[0] <= style <= self.style_range[1])
                
    def center(self) -> torch.Tensor:
        """Get the center point of the interval."""
        att_center = (self.att_range[0] + self.att_range[1]) / 2
        style_center = (self.style_range[0] + self.style_range[1]) / 2
        return torch.tensor([att_center, style_center])
        
    def width(self) -> Tuple[float, float]:
        """Get the width of the interval in each dimension."""
        att_width = self.att_range[1] - self.att_range[0]
        style_width = self.style_range[1] - self.style_range[0]
        return (att_width, style_width)
        
    def area(self) -> float:
        """Get the area of the interval."""
        att_width, style_width = self.width()
        return att_width * style_width
        
    def split(self) -> List['Interval']:
        """
        Split interval into four equal sub-intervals.
        
        Returns:
            List of four sub-intervals
        """
        att_mid = (self.att_range[0] + self.att_range[1]) / 2
        style_mid = (self.style_range[0] + self.style_range[1]) / 2
        
        return [
            Interval((self.att_range[0], att_mid), (self.style_range[0], style_mid)),
            Interval((att_mid, self.att_range[1]), (self.style_range[0], style_mid)),
            Interval((self.att_range[0], att_mid), (style_mid, self.style_range[1])),
            Interval((att_mid, self.att_range[1]), (style_mid, self.style_range[1]))
        ]
    
    def split_along_dimension(self, dimension: int) -> List['Interval']:
        """
        Split interval along a specific dimension.
        
        Args:
            dimension: Dimension to split (0 for attentiveness, 1 for driving style)
            
        Returns:
            List of two sub-intervals
        """
        if dimension == 0:  # Split along attentiveness
            att_mid = (self.att_range[0] + self.att_range[1]) / 2
            return [
                Interval((self.att_range[0], att_mid), self.style_range),
                Interval((att_mid, self.att_range[1]), self.style_range)
            ]
        else:  # Split along driving style
            style_mid = (self.style_range[0] + self.style_range[1]) / 2
            return [
                Interval(self.att_range, (self.style_range[0], style_mid)),
                Interval(self.att_range, (style_mid, self.style_range[1]))
            ]
        
    def uniform_sample(self, n: int = 1) -> torch.Tensor:
        """
        Generate uniform random samples from the interval.
        
        Args:
            n: Number of samples
            
        Returns:
            Tensor of shape [n, 2] with sampled internal states
        """
        att_samples = torch.rand(n) * (self.att_range[1] - self.att_range[0]) + self.att_range[0]
        style_samples = torch.rand(n) * (self.style_range[1] - self.style_range[0]) + self.style_range[0]
        
        return torch.stack([att_samples, style_samples], dim=1)


class IntervalBelief:
    """
    Represents a belief distribution over intervals in the internal state space.
    """
    
    def __init__(self, intervals: Optional[List[Interval]] = None):
        """
        Initialize belief over intervals.
        
        Args:
            intervals: List of intervals (default: single interval covering [0,1]×[0,1])
        """
        if intervals is None:
            # Default: single interval covering the entire space
            intervals = [Interval((0.0, 1.0), (0.0, 1.0))]
            
        self.intervals = intervals
        
        # Initialize uniform belief
        n_intervals = len(intervals)
        self.probs = torch.ones(n_intervals) / n_intervals
        
        # For tracking history
        self.history = []
        self.save_current_state()
        
    def __str__(self):
        result = "IntervalBelief:\n"
        for i, (interval, prob) in enumerate(zip(self.intervals, self.probs)):
            result += f"  {i}: {interval}, p={prob.item():.4f}\n"
        return result
        
    def update(self, 
            likelihood_fn: Callable[[torch.Tensor], torch.Tensor],
            num_samples: int = 10) -> None:
        """
        Update belief using Bayesian inference.
        
        Args:
            likelihood_fn: Function that computes likelihood of observation given internal state
            num_samples: Number of samples per interval for likelihood computation
        """
        # Compute likelihood for each interval using sampling
        interval_likelihoods = []
        
        for interval in self.intervals:
            # Sample points from the interval
            samples = interval.uniform_sample(num_samples)
            
            # Compute likelihood for each sample
            sample_likelihoods = []
            for sample in samples:
                likelihood = likelihood_fn(sample)
                sample_likelihoods.append(likelihood)
            
            # Average likelihood over the interval
            sample_likelihoods = torch.stack(sample_likelihoods)
            interval_likelihood = torch.mean(sample_likelihoods)
            
            # Ensure likelihood is non-zero to avoid numerical issues
            interval_likelihood = torch.max(interval_likelihood, torch.tensor(1e-10))
            
            interval_likelihoods.append(interval_likelihood)
                
        # Convert to tensor
        likelihoods = torch.stack(interval_likelihoods)
        
        # CRITICAL FIX: Ensure likelihoods vary across intervals
        # If all likelihoods are the same, add small random noise to create differences
        if torch.all(likelihoods == likelihoods[0]):
            print("WARNING: All likelihoods are identical, adding noise to differentiate")
            likelihoods = likelihoods + torch.rand_like(likelihoods) * 0.01
        
        # Bayesian update
        posterior = self.probs * likelihoods
        
        # Normalize
        posterior_sum = torch.sum(posterior)
        if posterior_sum > 0:
            self.probs = posterior / posterior_sum
        
        # Save updated state to history
        self.save_current_state()

    def refine(self, threshold: float = 0.5) -> None:
        """
        Refine intervals with high probability.
        
        Args:
            threshold: Probability threshold for refinement
        """
        new_intervals = []
        new_probs = []
        
        for interval, prob in zip(self.intervals, self.probs):
            if prob > threshold and interval.area() > 0.01:
                # Split interval
                sub_intervals = interval.split()
                
                # Distribute probability uniformly among sub-intervals
                for sub_interval in sub_intervals:
                    new_intervals.append(sub_interval)
                    new_probs.append(prob / len(sub_intervals))
            else:
                # Keep interval as is
                new_intervals.append(interval)
                new_probs.append(prob)
                
        # Update intervals and probabilities
        self.intervals = new_intervals
        self.probs = torch.tensor(new_probs)
        
        # Save updated state to history
        self.save_current_state()
        
    def refine_with_gradient(self, 
                           likelihood_fn: Callable[[torch.Tensor], torch.Tensor],
                           threshold: float = 0.3, 
                           num_samples: int = 20) -> None:
        """
        Refine intervals using gradient-based splitting.
        
        This method prioritizes splitting along dimensions that show 
        higher variability in likelihood.
        
        Args:
            likelihood_fn: Function that computes likelihood of observation given internal state
            threshold: Probability threshold for refinement
            num_samples: Number of samples for gradient estimation
        """
        new_intervals = []
        new_probs = []
        
        for interval, prob in zip(self.intervals, self.probs):
            if prob > threshold and interval.area() > 0.01:
                # Estimate gradient of likelihood along each dimension
                att_grad = self._estimate_gradient(interval, likelihood_fn, 0, num_samples)
                style_grad = self._estimate_gradient(interval, likelihood_fn, 1, num_samples)
                
                # Choose dimension with higher gradient magnitude
                if abs(att_grad) > abs(style_grad):
                    # Split along attentiveness
                    sub_intervals = interval.split_along_dimension(0)
                else:
                    # Split along driving style
                    sub_intervals = interval.split_along_dimension(1)
                
                # Compute likelihoods for each sub-interval to distribute probability
                sub_probs = []
                for sub_interval in sub_intervals:
                    samples = sub_interval.uniform_sample(num_samples)
                    sample_likelihoods = torch.stack([likelihood_fn(sample) for sample in samples])
                    sub_probs.append(torch.mean(sample_likelihoods).item())
                
                # Normalize sub-probabilities
                if sum(sub_probs) > 0:
                    sub_probs = [p / sum(sub_probs) for p in sub_probs]
                else:
                    sub_probs = [1.0 / len(sub_intervals)] * len(sub_intervals)
                
                # Distribute probability according to sub-interval likelihoods
                for sub_interval, sub_prob in zip(sub_intervals, sub_probs):
                    new_intervals.append(sub_interval)
                    new_probs.append(prob * sub_prob)
            else:
                # Keep interval as is
                new_intervals.append(interval)
                new_probs.append(prob)
                
        # Update intervals and probabilities
        self.intervals = new_intervals
        self.probs = torch.tensor(new_probs)
        
        # Save updated state to history
        self.save_current_state()
    
    def _estimate_gradient(self, 
                         interval: Interval, 
                         likelihood_fn: Callable[[torch.Tensor], torch.Tensor],
                         dimension: int, 
                         num_samples: int) -> float:
        """
        Estimate gradient of likelihood function along a specific dimension.
        
        Args:
            interval: Interval to estimate gradient for
            likelihood_fn: Function that computes likelihood of observation given internal state
            dimension: Dimension to estimate gradient (0 for attentiveness, 1 for driving style)
            num_samples: Number of samples for estimation
            
        Returns:
            Estimated gradient
        """
        # Generate samples with low and high values along the specified dimension
        if dimension == 0:  # Attentiveness
            low_points = torch.zeros(num_samples, 2)
            high_points = torch.zeros(num_samples, 2)
            
            # Set attentiveness to low and high values
            low_points[:, 0] = torch.ones(num_samples) * interval.att_range[0]
            high_points[:, 0] = torch.ones(num_samples) * interval.att_range[1]
            
            # Sample driving style uniformly
            style_samples = torch.rand(num_samples) * (interval.style_range[1] - interval.style_range[0]) + interval.style_range[0]
            low_points[:, 1] = style_samples
            high_points[:, 1] = style_samples
        else:  # Driving style
            low_points = torch.zeros(num_samples, 2)
            high_points = torch.zeros(num_samples, 2)
            
            # Sample attentiveness uniformly
            att_samples = torch.rand(num_samples) * (interval.att_range[1] - interval.att_range[0]) + interval.att_range[0]
            low_points[:, 0] = att_samples
            high_points[:, 0] = att_samples
            
            # Set driving style to low and high values
            low_points[:, 1] = torch.ones(num_samples) * interval.style_range[0]
            high_points[:, 1] = torch.ones(num_samples) * interval.style_range[1]
        
        # Compute likelihoods
        low_likelihoods = torch.stack([likelihood_fn(p) for p in low_points])
        high_likelihoods = torch.stack([likelihood_fn(p) for p in high_points])
        
        # Compute average gradient
        if dimension == 0:
            dim_width = interval.att_range[1] - interval.att_range[0]
        else:
            dim_width = interval.style_range[1] - interval.style_range[0]
            
        if dim_width == 0:
            return 0.0
            
        gradient = torch.mean(high_likelihoods - low_likelihoods) / dim_width
        
        return gradient.item()
        
    def entropy(self) -> torch.Tensor:
        """
        Compute entropy of the belief distribution.
        
        Returns:
            Entropy value
        """
        # Filter out zeros to avoid log(0)
        valid_probs = self.probs[self.probs > 0]
        
        # Compute entropy: -sum(p * log(p))
        return -torch.sum(valid_probs * torch.log(valid_probs))
        
    def max_probability_interval(self) -> Tuple[Interval, float]:
        """
        Get the interval with highest probability.
        
        Returns:
            (interval, probability) pair
        """
        max_idx = torch.argmax(self.probs).item()
        return self.intervals[max_idx], self.probs[max_idx].item()
        
    def expected_value(self) -> torch.Tensor:
        """
        Compute expected value of internal state.
        
        Returns:
            Expected internal state [attentiveness, driving_style]
        """
        # Compute weighted sum of interval centers
        centers = torch.stack([interval.center() for interval in self.intervals])
        return torch.sum(centers * self.probs.unsqueeze(1), dim=0)
    
    def compute_expected_value_with_uncertainty(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute expected value and uncertainty bounds.
        
        Returns:
            Tuple of (expected_value, lower_bound, upper_bound)
        """
        # Compute expected value
        expected = self.expected_value()
        
        # Initialize min and max bounds
        att_min, att_max = 1.0, 0.0
        style_min, style_max = 1.0, 0.0
        
        # Compute weighted bounds
        for interval, prob in zip(self.intervals, self.probs):
            if prob < 0.01:  # Skip intervals with very low probability
                continue
                
            # Update attentiveness bounds
            att_min = min(att_min, interval.att_range[0])
            att_max = max(att_max, interval.att_range[1])
            
            # Update driving style bounds
            style_min = min(style_min, interval.style_range[0])
            style_max = max(style_max, interval.style_range[1])
        
        # Create bound tensors
        lower_bound = torch.tensor([att_min, style_min])
        upper_bound = torch.tensor([att_max, style_max])
        
        return expected, lower_bound, upper_bound
        
    def sample(self, n: int = 1) -> torch.Tensor:
        """
        Sample internal states from the belief distribution.
        
        Args:
            n: Number of samples
            
        Returns:
            Tensor of shape [n, 2] with sampled internal states
        """
        # Sample interval indices based on probabilities
        interval_indices = torch.multinomial(self.probs, n, replacement=True)
        
        # Sample from selected intervals
        samples = []
        for idx in interval_indices:
            interval = self.intervals[idx]
            sample = interval.uniform_sample(1)[0]
            samples.append(sample)
            
        return torch.stack(samples)
    
    def sample_proportionally(self, total_samples: int) -> Dict[int, int]:
        """
        Distribute samples among intervals proportionally to probabilities.
        
        Args:
            total_samples: Total number of samples to allocate
            
        Returns:
            Dictionary mapping interval indices to number of samples
        """
        # CURRENT IMPLEMENTATION - allocates a min of 1 sample to each interval
        # We'll optimize to focus more precisely on the probability distribution
        
        # Initialize sample allocation
        sample_allocation = {}
        
        # Simple proportional allocation - no minimum samples per interval
        for i, interval in enumerate(self.intervals):
            # Allocate samples purely based on probability
            samples_for_interval = int(self.probs[i].item() * total_samples)
            
            # Store allocation if non-zero
            if samples_for_interval > 0:
                sample_allocation[i] = samples_for_interval
        
        # Check if we've allocated all samples
        allocated_samples = sum(sample_allocation.values())
        
        # If we have unallocated samples due to rounding, assign to highest probability intervals
        if allocated_samples < total_samples:
            # Sort intervals by probability (descending)
            remaining = total_samples - allocated_samples
            sorted_indices = torch.argsort(self.probs, descending=True)
            
            # Distribute remaining samples to highest probability intervals
            for idx in sorted_indices:
                if remaining <= 0:
                    break
                
                i = idx.item()
                if i in sample_allocation:
                    sample_allocation[i] += 1
                    remaining -= 1
                else:
                    # If this interval had 0 samples, give it one
                    sample_allocation[i] = 1
                    remaining -= 1
        
        # Ensure we're using exactly the requested number of samples
        assert sum(sample_allocation.values()) == total_samples, "Sample allocation must match total samples"
        
        return sample_allocation

    def save_current_state(self):
        """Save current belief state to history."""
        # Save a deep copy of intervals and probabilities
        current_state = {
            'intervals': self.intervals.copy(),
            'probs': self.probs.clone()
        }
        self.history.append(current_state)
    
    def plot(self, ax=None, title=None, show_center=False):
        """
        Plot the belief distribution over intervals.
        
        Args:
            ax: Matplotlib axis (creates new figure if None)
            title: Title for the plot
            show_center: Whether to show the expected value as a point
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        # Plot each interval as a rectangle
        for interval, prob in zip(self.intervals, self.probs):
            att_min, att_max = interval.att_range
            style_min, style_max = interval.style_range
            
            # Create rectangle with color based on probability
            rect = Rectangle(
                (att_min, style_min),
                att_max - att_min,
                style_max - style_min,
                alpha=prob.item(),  # Use probability for transparency
                facecolor='red',
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(rect)
            
            # Add text if probability is high enough
            if prob > 0.1:
                ax.text(
                    (att_min + att_max) / 2,
                    (style_min + style_max) / 2,
                    f"{prob.item():.2f}",
                    ha='center',
                    va='center',
                    color='white' if prob > 0.5 else 'black'
                )
        
        # Show expected value if requested
        if show_center:
            expected = self.expected_value()
            ax.plot(expected[0].item(), expected[1].item(), 'ko', markersize=8)
            
            # Also show uncertainty bounds
            expected, lower, upper = self.compute_expected_value_with_uncertainty()
            ax.axhspan(lower[1].item(), upper[1].item(), 
                      xmin=(lower[0].item()-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]), 
                      xmax=(upper[0].item()-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0]), 
                      alpha=0.2, color='blue')
                
        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Attentiveness')
        ax.set_ylabel('Driving Style (Aggressiveness)')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Belief Distribution over Internal States')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return ax
    
    def plot_history(self, true_state=None, figsize=(15, 10), max_plots=6):
        """
        Plot the history of belief updates.
        
        Args:
            true_state: True human internal state (optional)
            figsize: Figure size
            max_plots: Maximum number of history states to plot
            
        Returns:
            Matplotlib figure
        """
        # Determine how many history states to plot
        history_length = len(self.history)
        if history_length <= max_plots:
            indices_to_plot = range(history_length)
        else:
            # Sample indices evenly from history
            step = history_length / max_plots
            indices_to_plot = [int(i * step) for i in range(max_plots)]
            # Ensure the last state is included
            if indices_to_plot[-1] != history_length - 1:
                indices_to_plot[-1] = history_length - 1
        
        # Determine subplot grid size
        grid_size = int(np.ceil(np.sqrt(len(indices_to_plot))))
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        if grid_size == 1:
            axes = np.array([[axes]])
        elif grid_size > 1 and len(axes.shape) == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each selected history state
        for i, idx in enumerate(indices_to_plot):
            row, col = i // grid_size, i % grid_size
            ax = axes[row, col]
            
            # Get state from history
            state = self.history[idx]
            
            # Create temporary belief for plotting
            temp_belief = IntervalBelief(state['intervals'])
            temp_belief.probs = state['probs']
            
            # Plot belief
            temp_belief.plot(ax, title=f"Step {idx}", show_center=True)
            
            # Add true state if provided
            if true_state is not None:
                ax.plot(true_state[0], true_state[1], 'g*', markersize=12, label='True State')
                ax.legend(loc='upper right')
        
        # Hide empty subplots
        for i in range(len(indices_to_plot), grid_size*grid_size):
            row, col = i // grid_size, i % grid_size
            axes[row, col].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig

class BinaryBelief:
    """
    Binary belief distribution over human internal state.
    Represents belief over two states: distracted (0) and attentive (1).
    """
    
    def __init__(self, p_attentive: float = 0.5):
        """
        Initialize binary belief.
        
        Args:
            p_attentive: Initial probability of human being attentive (default: 0.5)
        """
        self.p_attentive = torch.tensor(p_attentive, dtype=torch.float32)
        self.p_distracted = 1.0 - self.p_attentive
        
        # For tracking history
        self.history = []
        self.save_current_state()
        
    def __str__(self):
        return f"BinaryBelief: P(attentive)={self.p_attentive.item():.4f}, P(distracted)={self.p_distracted.item():.4f}"
    
    def update(self, likelihood_attentive: float, likelihood_distracted: float):
        """
        Update belief using Bayesian inference.
        
        Args:
            likelihood_attentive: P(observation | attentive)
            likelihood_distracted: P(observation | distracted)
        """
        # Convert to tensors if needed
        if not isinstance(likelihood_attentive, torch.Tensor):
            likelihood_attentive = torch.tensor(likelihood_attentive, dtype=torch.float32)
        if not isinstance(likelihood_distracted, torch.Tensor):
            likelihood_distracted = torch.tensor(likelihood_distracted, dtype=torch.float32)
            
        # Bayesian update
        posterior_attentive = self.p_attentive * likelihood_attentive
        posterior_distracted = self.p_distracted * likelihood_distracted
        
        # Normalize
        total = posterior_attentive + posterior_distracted
        if total > 0:
            self.p_attentive = posterior_attentive / total
            self.p_distracted = posterior_distracted / total
        
        # Save state to history
        self.save_current_state()
    
    def update_with_likelihood_fn(self, likelihood_fn: Callable[[torch.Tensor], torch.Tensor]):
        """
        Update belief using a likelihood function.
        
        Args:
            likelihood_fn: Function that computes likelihood given internal state
        """
        # Compute likelihoods for both states
        phi_distracted = torch.tensor([0.0], dtype=torch.float32)  # Binary: 0 for distracted
        phi_attentive = torch.tensor([1.0], dtype=torch.float32)   # Binary: 1 for attentive
        
        likelihood_distracted = likelihood_fn(phi_distracted)
        likelihood_attentive = likelihood_fn(phi_attentive)
        
        # Update using computed likelihoods
        self.update(likelihood_attentive.item(), likelihood_distracted.item())
    
    def entropy(self) -> torch.Tensor:
        """
        Compute binary entropy of the belief distribution.
        
        Returns:
            Binary entropy value
        """
        # Handle edge cases
        if self.p_attentive <= 0 or self.p_attentive >= 1:
            return torch.tensor(0.0, dtype=torch.float32)
        
        # Binary entropy: -p*log(p) - (1-p)*log(1-p)
        return -(self.p_attentive * torch.log(self.p_attentive) + 
                self.p_distracted * torch.log(self.p_distracted))
    
    def sample(self, n: int = 1) -> torch.Tensor:
        """
        Sample internal states from the belief distribution.
        
        Args:
            n: Number of samples
            
        Returns:
            Tensor of shape [n] with sampled internal states (0 or 1)
        """
        # Sample from Bernoulli distribution
        samples = torch.bernoulli(self.p_attentive.repeat(n))
        return samples
    
    def get_probabilities(self) -> Tuple[float, float]:
        """
        Get the current probabilities.
        
        Returns:
            Tuple of (p_distracted, p_attentive)
        """
        return (self.p_distracted.item(), self.p_attentive.item())
    
    def get_most_likely_state(self) -> int:
        """
        Get the most likely internal state.
        
        Returns:
            0 if distracted is more likely, 1 if attentive is more likely
        """
        return 1 if self.p_attentive > 0.5 else 0
    
    def save_current_state(self):
        """Save current belief state to history."""
        current_state = {
            'p_attentive': self.p_attentive.clone(),
            'p_distracted': self.p_distracted.clone()
        }
        self.history.append(current_state)
    
    def plot(self, ax=None, title=None):
        """
        Plot the binary belief distribution as a bar chart.
        
        Args:
            ax: Matplotlib axis (creates new figure if None)
            title: Title for the plot
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create bar chart
        states = ['Distracted', 'Attentive']
        probabilities = [self.p_distracted.item(), self.p_attentive.item()]
        colors = ['red', 'green']
        
        bars = ax.bar(states, probabilities, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.3f}', ha='center', va='bottom', fontsize=12)
        
        # Set axis properties
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Probability')
        ax.set_xlabel('Human Internal State')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Binary Belief Distribution')
        
        # Add entropy information
        entropy_val = self.entropy().item()
        ax.text(0.5, 0.95, f'Entropy: {entropy_val:.3f}', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add grid
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        return ax
    
    def plot_history(self, ax=None, true_state=None):
        """
        Plot the history of belief updates.
        
        Args:
            ax: Matplotlib axis (creates new figure if None)
            true_state: True human internal state (0 or 1, optional)
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if not self.history:
            ax.text(0.5, 0.5, "No history data available",
                   ha='center', va='center', transform=ax.transAxes)
            return ax
        
        # Extract probabilities from history
        time_steps = list(range(len(self.history)))
        p_attentive_history = [state['p_attentive'].item() for state in self.history]
        p_distracted_history = [state['p_distracted'].item() for state in self.history]
        
        # Plot probabilities over time
        ax.plot(time_steps, p_attentive_history, 'g-', linewidth=2, label='P(Attentive)')
        ax.plot(time_steps, p_distracted_history, 'r-', linewidth=2, label='P(Distracted)')
        
        # Add true state if provided
        if true_state is not None:
            true_prob = 1.0 if true_state == 1 else 0.0
            ax.axhline(y=true_prob, color='black', linestyle='--', alpha=0.7,
                      label=f'True State: {"Attentive" if true_state == 1 else "Distracted"}')
        
        # Set axis properties
        ax.set_xlim(0, len(time_steps) - 1)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Probability')
        ax.set_title('Binary Belief Evolution Over Time')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return ax
    
    def plot_entropy_history(self, ax=None):
        """
        Plot the history of entropy values.
        
        Args:
            ax: Matplotlib axis (creates new figure if None)
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if not self.history:
            ax.text(0.5, 0.5, "No history data available",
                   ha='center', va='center', transform=ax.transAxes)
            return ax
        
        # Compute entropy for each historical state
        entropy_history = []
        for state in self.history:
            p_att = state['p_attentive']
            if p_att <= 0 or p_att >= 1:
                entropy = 0.0
            else:
                p_dis = state['p_distracted']
                entropy = -(p_att * torch.log(p_att) + p_dis * torch.log(p_dis)).item()
            entropy_history.append(entropy)
        
        # Plot entropy over time
        time_steps = list(range(len(self.history)))
        ax.plot(time_steps, entropy_history, 'b-', linewidth=2, marker='o')
        
        # Add maximum entropy line
        max_entropy = -2 * (0.5 * np.log(0.5))  # Binary entropy is maximized at p=0.5
        ax.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.5,
                  label=f'Max Entropy: {max_entropy:.3f}')
        
        # Set axis properties
        ax.set_xlim(0, len(time_steps) - 1)
        ax.set_ylim(0, max_entropy * 1.1)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Entropy')
        ax.set_title('Binary Belief Entropy Over Time')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        return ax

if __name__ == "__main__":
    # Test the belief representation
    
    # Create initial belief
    belief = IntervalBelief()
    print("Initial belief:")
    print(belief)
    
    # Define a likelihood function that favors attentive drivers
    def attentive_likelihood(phi):
        attentiveness = phi[0]
        return torch.tensor(attentiveness)  # Higher attentiveness -> higher likelihood
    
    # Update belief
    belief.update(attentive_likelihood)
    print("\nBelief after update:")
    print(belief)
    
    # Refine intervals with gradient
    belief.refine_with_gradient(attentive_likelihood, threshold=0.3)
    print("\nBelief after gradient-based refinement:")
    print(belief)
    
    # Compute entropy
    entropy = belief.entropy()
    print(f"\nEntropy: {entropy.item()}")
    
    # Get maximum probability interval
    max_interval, max_prob = belief.max_probability_interval()
    print(f"\nMax probability interval: {max_interval}, p={max_prob}")
    
    # Get expected value with uncertainty
    expected, lower, upper = belief.compute_expected_value_with_uncertainty()
    print(f"\nExpected internal state: {expected}")
    print(f"Lower bound: {lower}")
    print(f"Upper bound: {upper}")
    
    # Test sample allocation
    sample_allocation = belief.sample_proportionally(100)
    print(f"\nSample allocation for 100 samples: {sample_allocation}")
    
    # Plot belief
    plt.figure(figsize=(10, 8))
    ax = belief.plot(show_center=True)
    plt.savefig("belief_visualization.png")
    
    # Create and update belief multiple times for history test
    test_belief = IntervalBelief()
    for i in range(6):
        # Define a parameterized likelihood function
        def likelihood_fn(phi, i=i):
            att_target = 0.2 + i * 0.15  # Moving target
            style_target = 0.3 + i * 0.1
            dist = torch.sqrt((phi[0] - att_target)**2 + (phi[1] - style_target)**2)
            return torch.exp(-5.0 * dist)
        
        # Update and refine
        test_belief.update(lambda phi: likelihood_fn(phi))
        # if i % 2 == 0:
        #     test_belief.refine(threshold=0.5)
        # else:
        #     test_belief.refine_with_gradient(lambda phi: likelihood_fn(phi), threshold=0.3)
        test_belief.refine(threshold=0.5)
    
    # Plot history
    true_state = torch.tensor([0.8, 0.7])
    history_fig = test_belief.plot_history(true_state=true_state)
    plt.savefig("belief_history.png")

    # Test binary belief
    print("\n\n=== Testing Binary Belief ===")
    
    # Create binary belief
    binary_belief = BinaryBelief(p_attentive=0.5)
    print("Initial binary belief:")
    print(binary_belief)
    
    # Test update with specific likelihoods
    binary_belief.update(likelihood_attentive=0.8, likelihood_distracted=0.2)
    print("\nAfter observing behavior more consistent with attentive:")
    print(binary_belief)
    
    # Test update with likelihood function
    def test_likelihood_fn(phi):
        # Higher likelihood for attentive state
        if phi[0] > 0.5:  # Attentive
            return torch.tensor(0.9)
        else:  # Distracted
            return torch.tensor(0.1)
    
    binary_belief.update_with_likelihood_fn(test_likelihood_fn)
    print("\nAfter another update favoring attentive:")
    print(binary_belief)
    
    # Test entropy
    print(f"\nEntropy: {binary_belief.entropy().item():.4f}")
    
    # Test sampling
    samples = binary_belief.sample(10)
    print(f"\nSampled states: {samples}")
    print(f"Fraction attentive in samples: {samples.mean().item():.2f}")
    
    # Plot binary belief
    plt.figure(figsize=(15, 5))
    
    # Current belief
    ax1 = plt.subplot(131)
    binary_belief.plot(ax=ax1)
    
    # History
    ax2 = plt.subplot(132)
    binary_belief.plot_history(ax=ax2, true_state=1)  # Assume true state is attentive
    
    # Entropy history
    ax3 = plt.subplot(133)
    binary_belief.plot_entropy_history(ax=ax3)
    
    plt.tight_layout()
    plt.savefig("binary_belief_test.png")
    plt.close()
    
    print("\nBinary belief visualization saved to binary_belief_test.png")