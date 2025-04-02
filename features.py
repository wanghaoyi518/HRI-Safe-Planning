import torch
import numpy as np
from typing import Callable, List, Tuple, Union, Optional

class Feature:
    """
    Feature class for computing reward components.
    
    A feature takes state and control inputs and produces a scalar value
    that represents a component of the reward function.
    """
    
    def __init__(self, f: Callable):
        """
        Initialize feature with a computing function.
        
        Args:
            f: Function that computes the feature value
        """
        self.f = f
        
    def __call__(self, t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute feature value.
        
        Args:
            t: Time step
            x: State vector
            u: Control vector
            
        Returns:
            Feature value as a scalar tensor
        """
        return self.f(t, x, u)


# def speed_feature(target_speed: float = 1.0, weight: float = 1.0) -> Feature:
#     """
#     Create a feature that rewards being close to target speed.
    
#     Args:
#         target_speed: Desired speed
#         weight: Scaling factor for the feature
        
#     Returns:
#         Feature that rewards being close to target speed
#     """
#     def compute(t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
#         # Assume x[3] is the velocity
#         speed_diff = (x[3] - target_speed)
#         return -weight * (speed_diff * speed_diff)
    
#     return Feature(compute)


# def lane_keeping_feature(lane_center: List[float], 
#                         lane_width: float,
#                         weight: float = 1.0) -> Feature:
#     """
#     Create a feature that rewards staying close to lane center.
    
#     Args:
#         lane_center: Center line of the lane [a, b, c] for ax + by + c = 0
#         lane_width: Width of the lane
#         weight: Scaling factor for the feature
        
#     Returns:
#         Feature that rewards staying close to lane center
#     """
#     a, b, c = lane_center
#     norm = np.sqrt(a**2 + b**2)
    
#     def compute(t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
#         # Compute perpendicular distance to line
#         dist = torch.abs(a*x[0] + b*x[1] + c) / norm
        
#         # Normalize by lane width
#         normalized_dist = dist / (lane_width/2)
        
#         # Quadratic penalty for deviation from center
#         return -weight * (normalized_dist * normalized_dist)
    
#     return Feature(compute)


# def collision_avoidance_feature(other_position: Callable[[int], torch.Tensor], 
#                               min_distance: float = 0.2,
#                               weight: float = 10.0) -> Feature:
#     """
#     Create a feature that penalizes getting too close to other agents.
    
#     Args:
#         other_position: Function that returns position of other agent at time t
#         min_distance: Minimum safe distance
#         weight: Scaling factor for the feature
        
#     Returns:
#         Feature that penalizes getting too close to other agents
#     """
#     def compute(t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
#         # Get other agent's position
#         other_pos = other_position(t)
        
#         # Compute Euclidean distance between positions
#         dx = x[0] - other_pos[0]
#         dy = x[1] - other_pos[1]
#         dist = torch.sqrt(dx*dx + dy*dy)
        
#         # Penalty increases as distance decreases below threshold
#         if dist > min_distance:
#             return torch.tensor(0.0, dtype=torch.float32)
#         else:
#             # Exponential penalty for getting too close
#             return -weight * torch.exp((min_distance - dist) / 0.05)
    
#     return Feature(compute)


# def control_smoothness_feature(bounds: List[Tuple[float, float]], 
#                              weight: float = 0.5) -> Feature:
#     """
#     Create a feature that penalizes extreme control inputs.
    
#     Args:
#         bounds: List of (min, max) tuples for each control dimension
#         weight: Scaling factor for the feature
        
#     Returns:
#         Feature that penalizes extreme control inputs
#     """
#     def compute(t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
#         penalty = torch.tensor(0.0, dtype=torch.float32)
        
#         for i, (min_val, max_val) in enumerate(bounds):
#             # Normalize control to [-1, 1] range based on bounds
#             range_size = max_val - min_val
#             mid_point = (max_val + min_val) / 2
#             normalized_u = (u[i] - mid_point) / (range_size / 2)
            
#             # Quadratic penalty that increases as controls approach boundaries
#             penalty = penalty - weight * (normalized_u * normalized_u)
            
#         return penalty
    
#     return Feature(compute)

# features.py - Add to existing file

# Add these IRL-specific feature functions

def speed_preference_feature(target_speed: float = 1.0):
    """
    Create a feature that rewards driving at a target speed.
    For IRL training.
    
    Args:
        target_speed: Desired speed
        
    Returns:
        Feature function for speed preference
    """
    def compute(t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        current_speed = x[3]  # Assuming state = [x, y, theta, v]
        speed_diff = current_speed - target_speed
        return -speed_diff**2
    
    return Feature(compute)

def lane_following_feature(lane_center: float = 0.0):
    """
    Create a feature that rewards staying in lane.
    For IRL training.
    
    Args:
        lane_center: Center position of the lane
        
    Returns:
        Feature function for lane following
    """
    def compute(t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        position = x[0]  # Assuming state = [x, y, theta, v]
        lane_deviation = position - lane_center
        return -lane_deviation**2
    
    return Feature(compute)

def obstacle_avoidance_feature(obstacle_position: torch.Tensor = None, safety_threshold: float = 0.2):
    """
    Create a feature that rewards avoiding obstacles.
    For IRL training.
    
    Args:
        obstacle_position: Position of obstacle to avoid
        safety_threshold: Minimum safe distance
        
    Returns:
        Feature function for obstacle avoidance
    """
    def compute(t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if obstacle_position is None:
            return torch.tensor(0.0)
            
        position = x[:2]  # Extract position [x, y]
        
        # Compute distance to obstacle
        distance = torch.norm(position - obstacle_position)
        
        if distance > safety_threshold:
            return torch.tensor(0.0)
        else:
            # Exponential penalty for getting too close
            return -torch.exp((safety_threshold - distance) / 0.05)
    
    return Feature(compute)

def control_smoothness_feature():
    """
    Create a feature that rewards smooth control inputs.
    For IRL training.
    
    Returns:
        Feature function for control smoothness
    """
    def compute(t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Penalize large control inputs
        return -(u[0]**2 + u[1]**2)
    
    return Feature(compute)

# Then in our IRL implementation, we would use these as:
# features = [
#     speed_preference_feature(1.0),
#     lane_following_feature(0.0),
#     obstacle_avoidance_feature(),
#     control_smoothness_feature()
# ]


def road_boundary_feature(boundaries: List[List[float]], 
                        margin: float = 0.1,
                        weight: float = 50.0) -> Feature:
    """
    Create a feature that penalizes getting too close to road boundaries.
    
    Args:
        boundaries: List of boundary lines [a, b, c] for ax + by + c = 0
        margin: Safety margin from boundary
        weight: Scaling factor for the feature
        
    Returns:
        Feature that penalizes getting too close to road boundaries
    """
    def compute(t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        penalty = torch.tensor(0.0, dtype=torch.float32)
        
        for a, b, c in boundaries:
            norm = np.sqrt(a**2 + b**2)
            # Compute perpendicular distance to boundary
            dist = torch.abs(a*x[0] + b*x[1] + c) / norm
            
            # Apply penalty if within margin
            if dist < margin:
                penalty = penalty - weight * torch.exp((margin - dist) / 0.05)
                
        return penalty
    
    return Feature(compute)


def gaussian_feature(mean: torch.Tensor, 
                   sigma: torch.Tensor,
                   weight: float = 1.0) -> Feature:
    """
    Create a Gaussian feature that is highest at the mean and falls off with distance.
    
    Args:
        mean: Center of the Gaussian [x, y]
        sigma: Standard deviation (or covariance matrix)
        weight: Scaling factor for the feature
        
    Returns:
        Feature with Gaussian distribution
    """
    def compute(t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Extract position from state
        pos = x[:2]
        
        # Compute squared Mahalanobis distance
        diff = pos - mean
        
        if isinstance(sigma, torch.Tensor) and sigma.dim() == 2:
            # Case: sigma is a covariance matrix
            inv_sigma = torch.inverse(sigma)
            dist_sq = torch.sum(diff * torch.mv(inv_sigma, diff))
        else:
            # Case: sigma is a scalar or diagonal vector
            dist_sq = torch.sum((diff / sigma) ** 2)
            
        # Compute Gaussian: exp(-0.5 * dist_sq)
        return weight * torch.exp(-0.5 * dist_sq)
    
    return Feature(compute)


def create_attentiveness_features(
    baseline_features: List[Tuple[Feature, float]],
    attention_level: float
) -> List[Tuple[Feature, float]]:
    """
    Adjust feature weights based on attentiveness level.
    
    Args:
        baseline_features: List of (feature, weight) pairs
        attention_level: Attentiveness level [0, 1]
        
    Returns:
        Adjusted list of (feature, weight) pairs
    """
    adjusted_features = []
    
    for feature, weight in baseline_features:
        # Adjust weights based on attentiveness
        if "collision" in str(feature):
            # More attentive drivers prioritize collision avoidance
            new_weight = weight * (0.5 + 0.5 * attention_level)
        elif "road_boundary" in str(feature):
            # More attentive drivers are more aware of boundaries
            new_weight = weight * (0.5 + 0.5 * attention_level)
        else:
            # Keep other weights the same
            new_weight = weight
            
        adjusted_features.append((feature, new_weight))
        
    return adjusted_features


def create_driving_style_features(
    baseline_features: List[Tuple[Feature, float]],
    aggressiveness: float
) -> List[Tuple[Feature, float]]:
    """
    Adjust feature weights based on driving style (aggressiveness).
    
    Args:
        baseline_features: List of (feature, weight) pairs
        aggressiveness: Aggressiveness level [0, 1]
        
    Returns:
        Adjusted list of (feature, weight) pairs
    """
    adjusted_features = []
    
    for feature, weight in baseline_features:
        # Adjust weights based on driving style
        if "speed" in str(feature):
            # More aggressive drivers prefer higher speeds
            new_weight = weight * (0.5 + 0.5 * aggressiveness)
        elif "control_smoothness" in str(feature):
            # More aggressive drivers use sharper controls
            new_weight = weight * (1.0 - 0.5 * aggressiveness)
        elif "collision" in str(feature):
            # More aggressive drivers accept closer distances
            new_weight = weight * (1.0 - 0.5 * aggressiveness)
        else:
            # Keep other weights the same
            new_weight = weight
            
        adjusted_features.append((feature, new_weight))
        
    return adjusted_features

def create_goal_reaching_feature(goal_position, weight=10.0):
    """
    Create a feature for reaching a goal position.
    
    Args:
        goal_position: Target position [x, y]
        weight: Scaling factor for the feature
        
    Returns:
        Feature for goal reaching
    """
    # Convert goal position to tensor if it isn't already
    if not isinstance(goal_position, torch.Tensor):
        goal_position = torch.tensor(goal_position, dtype=torch.float32)
    
    def compute(t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Extract robot position from state
        robot_pos = x[:2]
        
        # Compute squared distance to goal
        dist = torch.sum((robot_pos - goal_position) ** 2)
        
        # Reward decreases with distance
        return -weight * dist
    
    return Feature(compute)
