import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
import matplotlib.pyplot as plt


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
            sample_likelihoods = torch.stack([likelihood_fn(sample) for sample in samples])
            
            # Average likelihood over the interval
            interval_likelihood = torch.mean(sample_likelihoods)
            interval_likelihoods.append(interval_likelihood)
            
        # Convert to tensor
        likelihoods = torch.stack(interval_likelihoods)
        
        # Bayesian update
        posterior = self.probs * likelihoods
        
        # Normalize
        posterior_sum = torch.sum(posterior)
        if posterior_sum > 0:
            self.probs = posterior / posterior_sum
        
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
    
    def plot(self, ax=None):
        """
        Plot the belief distribution over intervals.
        
        Args:
            ax: Matplotlib axis (creates new figure if None)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        # Plot each interval as a rectangle
        for interval, prob in zip(self.intervals, self.probs):
            att_min, att_max = interval.att_range
            style_min, style_max = interval.style_range
            
            # Create rectangle with color based on probability
            rect = plt.Rectangle(
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
                
        # Set axis properties
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Attentiveness')
        ax.set_ylabel('Driving Style (Aggressiveness)')
        ax.set_title('Belief Distribution over Internal States')
        
        # Add grid
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
    
    # Refine intervals
    belief.refine(threshold=0.3)
    print("\nBelief after refinement:")
    print(belief)
    
    # Compute entropy
    entropy = belief.entropy()
    print(f"\nEntropy: {entropy.item()}")
    
    # Get maximum probability interval
    max_interval, max_prob = belief.max_probability_interval()
    print(f"\nMax probability interval: {max_interval}, p={max_prob}")
    
    # Get expected value
    expected_value = belief.expected_value()
    print(f"\nExpected internal state: {expected_value}")
    
    # Plot belief
    belief.plot()
    plt.show()