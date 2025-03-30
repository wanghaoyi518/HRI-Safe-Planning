import torch
import numpy as np
from typing import List, Dict, Callable, Tuple, Union, Optional
from features import Feature, speed_feature, lane_keeping_feature, collision_avoidance_feature, control_smoothness_feature, road_boundary_feature


class Reward:
    """
    Reward class that combines multiple weighted features.
    Used to define reward functions for both human and robot agents.
    """
    
    def __init__(self, features: List[Tuple[Feature, float]] = None, name: str = "unnamed"):
        """
        Initialize reward with a list of (feature, weight) pairs.
        
        Args:
            features: List of (feature, weight) pairs
            name: Name for the reward function
        """
        self.features = features or []
        self.name = name
        
    def add_feature(self, feature: Feature, weight: float = 1.0):
        """
        Add a feature to the reward function.
        
        Args:
            feature: Feature to add
            weight: Weight for the feature
        """
        self.features.append((feature, weight))
        
    def __call__(self, t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute total reward.
        
        Args:
            t: Time step
            x: State vector
            u: Control vector
            
        Returns:
            Total reward as a scalar tensor
        """
        total_reward = torch.tensor(0.0, dtype=torch.float32)
        
        for feature, weight in self.features:
            feature_value = feature(t, x, u)
            total_reward = total_reward + weight * feature_value
            
        return total_reward


class ParameterizedReward(Reward):
    """
    Reward function parameterized by human internal state.
    Adapts feature weights based on attentiveness and driving style.
    """
    
    def __init__(self, 
                base_features: List[Tuple[Feature, float]],
                name: str = "parameterized"):
        """
        Initialize parameterized reward.
        
        Args:
            base_features: Base feature set with default weights
            name: Name for the reward function
        """
        super().__init__(features=base_features, name=name)
        self.base_features = base_features
        
    def __call__(self, t: int, x: torch.Tensor, u: torch.Tensor, phi: torch.Tensor = None) -> torch.Tensor:
        """
        Compute reward based on internal state.
        
        Args:
            t: Time step
            x: State vector
            u: Control vector
            phi: Internal state vector [attentiveness, driving_style]
            
        Returns:
            Total reward as a scalar tensor
        """
        if phi is None:
            # Use base features if no internal state is provided
            return super().__call__(t, x, u)
        
        # Extract attentiveness and driving style
        attentiveness = phi[0].item()
        driving_style = phi[1].item()
        
        # Compute adaptive weights
        total_reward = torch.tensor(0.0, dtype=torch.float32)
        
        for i, (feature, base_weight) in enumerate(self.base_features):
            # Adjust weight based on feature type and internal state
            weight = self._adjust_weight(feature, base_weight, attentiveness, driving_style)
            feature_value = feature(t, x, u)
            total_reward = total_reward + weight * feature_value
            
        return total_reward
    
    def _adjust_weight(self, feature, base_weight, attentiveness, driving_style):
        """
        Adjust feature weight based on internal state.
        
        Args:
            feature: Feature object
            base_weight: Base weight for the feature
            attentiveness: Attentiveness level [0, 1]
            driving_style: Driving style (aggressiveness) [0, 1]
            
        Returns:
            Adjusted weight
        """
        feature_name = feature.f.__name__ if hasattr(feature.f, "__name__") else str(feature)
        
        # Adjust weights based on feature type
        if "collision" in feature_name:
            # Attentive drivers prioritize collision avoidance
            # Conservative drivers prioritize safety
            att_factor = 0.5 + 0.5 * attentiveness
            style_factor = 1.0 - 0.5 * driving_style
            return base_weight * att_factor * style_factor
            
        elif "speed" in feature_name:
            # Aggressive drivers value speed more
            style_factor = 0.5 + 0.5 * driving_style
            return base_weight * style_factor
            
        elif "lane" in feature_name:
            # Attentive drivers stay in lanes better
            att_factor = 0.7 + 0.3 * attentiveness
            return base_weight * att_factor
            
        elif "control" in feature_name:
            # Aggressive drivers use sharper controls
            style_factor = 1.0 - 0.5 * driving_style
            return base_weight * style_factor
            
        elif "boundary" in feature_name:
            # Attentive drivers are more aware of boundaries
            att_factor = 0.5 + 0.5 * attentiveness
            return base_weight * att_factor
            
        # Default: no adjustment
        return base_weight


class HumanReward(ParameterizedReward):
    """
    Specific reward function for human drivers.
    """
    
    def __init__(self):
        """Initialize with typical human driving preferences."""
        # Define base features for human driving
        base_features = [
            (speed_feature(target_speed=1.0), 10.0),                    # Prefer driving at target speed
            (lane_keeping_feature([0, 1, 0], 0.1), 1.0),                # Stay in lane
            (control_smoothness_feature([(-1, 1), (-1, 2)]), 5.0),      # Smooth control inputs
            (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 50.0)      # Avoid road boundaries
        ]
        
        super().__init__(base_features=base_features, name="human_reward")


class RobotReward(Reward):
    """
    Reward function for robot planning.
    Includes information gain component for active learning.
    """
    
    def __init__(self, task_features: List[Tuple[Feature, float]] = None, 
                info_gain_weight: float = 1.0):
        """
        Initialize robot reward.
        
        Args:
            task_features: Features for the robot's task
            info_gain_weight: Weight for information gain component
        """
        super().__init__(features=task_features or [], name="robot_reward")
        self.info_gain_weight = info_gain_weight
        self.belief_entropy_prev = None
        
    def __call__(self, t: int, x: torch.Tensor, u: torch.Tensor, 
                belief_entropy: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute robot reward including information gain.
        
        Args:
            t: Time step
            x: State vector
            u: Control vector
            belief_entropy: Current entropy of belief distribution
            
        Returns:
            Total reward including task reward and information gain
        """
        # Compute task reward
        task_reward = super().__call__(t, x, u)
        
        # Add information gain if belief entropy is provided
        if belief_entropy is not None:
            if self.belief_entropy_prev is None:
                self.belief_entropy_prev = belief_entropy
                info_gain = torch.tensor(0.0, dtype=torch.float32)
            else:
                # Information gain is the reduction in entropy
                info_gain = self.belief_entropy_prev - belief_entropy
                self.belief_entropy_prev = belief_entropy
                
            # Add information gain to reward
            return task_reward + self.info_gain_weight * info_gain
        
        return task_reward


# Pre-defined reward functions for common driving styles
def create_normal_reward() -> HumanReward:
    """Create a reward function for normal driving style."""
    reward = HumanReward()
    return reward


def create_aggressive_reward() -> HumanReward:
    """Create a reward function for aggressive driving style."""
    reward = HumanReward()
    # Override with more aggressive weights
    reward.base_features = [
        (speed_feature(target_speed=1.5), 15.0),                    # Higher target speed
        (lane_keeping_feature([0, 1, 0], 0.1), 0.8),                # Less emphasis on lane keeping
        (control_smoothness_feature([(-1, 1), (-1, 2)]), 3.0),      # Less emphasis on smooth controls
        (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 45.0)      # Slightly less concern for boundaries
    ]
    return reward


def create_conservative_reward() -> HumanReward:
    """Create a reward function for conservative driving style."""
    reward = HumanReward()
    # Override with more conservative weights
    reward.base_features = [
        (speed_feature(target_speed=0.7), 8.0),                     # Lower target speed
        (lane_keeping_feature([0, 1, 0], 0.1), 1.5),                # More emphasis on lane keeping
        (control_smoothness_feature([(-1, 1), (-1, 2)]), 7.0),      # More emphasis on smooth controls
        (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 60.0)      # More concern for boundaries
    ]
    return reward


def create_attentive_reward() -> HumanReward:
    """Create a reward function for attentive driving."""
    reward = HumanReward()
    # Override with attentive weights
    reward.base_features = [
        (speed_feature(target_speed=1.0), 10.0),                    # Normal target speed
        (lane_keeping_feature([0, 1, 0], 0.1), 1.3),                # Good lane keeping
        (control_smoothness_feature([(-1, 1), (-1, 2)]), 5.0),      # Normal control smoothness
        (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 55.0)      # Higher concern for boundaries
    ]
    return reward


def create_distracted_reward() -> HumanReward:
    """Create a reward function for distracted driving."""
    reward = HumanReward()
    # Override with distracted weights
    reward.base_features = [
        (speed_feature(target_speed=1.0), 10.0),                    # Normal target speed
        (lane_keeping_feature([0, 1, 0], 0.1), 0.7),                # Poorer lane keeping
        (control_smoothness_feature([(-1, 1), (-1, 2)]), 4.0),      # Less smooth controls
        (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 35.0)      # Less concern for boundaries
    ]
    return reward


def create_robot_reward(info_gain_weight: float = 1.0) -> RobotReward:
    """
    Create a default reward function for the robot.
    
    Args:
        info_gain_weight: Weight for information gain component
        
    Returns:
        Robot reward function
    """
    # Define base features for robot driving
    task_features = [
        (speed_feature(target_speed=0.8), 8.0),                     # Slightly conservative speed
        (lane_keeping_feature([0, 1, 0], 0.1), 1.2),                # Good lane keeping
        (control_smoothness_feature([(-1, 1), (-1, 2)]), 6.0),      # Smooth controls
        (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 55.0)      # High concern for boundaries
    ]
    
    return RobotReward(task_features=task_features, info_gain_weight=info_gain_weight)


if __name__ == "__main__":
    # Test reward functions
    
    # Create test state and control
    x = torch.tensor([0.0, 0.0, 0.0, 1.2])  # [x, y, theta, v]
    u = torch.tensor([0.1, 0.5])            # [steering, acceleration]
    
    # Test human reward
    human_reward = create_normal_reward()
    reward_value = human_reward(0, x, u)
    print(f"Normal human reward: {reward_value.item()}")
    
    # Test with different internal states
    attentive_aggressive = torch.tensor([0.9, 0.8])  # High attentiveness, high aggressiveness
    distracted_conservative = torch.tensor([0.2, 0.3])  # Low attentiveness, low aggressiveness
    
    reward_att_agg = human_reward(0, x, u, attentive_aggressive)
    reward_dis_con = human_reward(0, x, u, distracted_conservative)
    
    print(f"Reward (attentive & aggressive): {reward_att_agg.item()}")
    print(f"Reward (distracted & conservative): {reward_dis_con.item()}")
    
    # Test robot reward with information gain
    robot_reward = create_robot_reward(info_gain_weight=2.0)
    task_reward = robot_reward(0, x, u)
    print(f"Robot task reward: {task_reward.item()}")
    
    # Test with information gain
    entropy_before = torch.tensor(1.5)  # Higher entropy (more uncertainty)
    entropy_after = torch.tensor(0.8)   # Lower entropy (less uncertainty)
    
    robot_reward_with_info = robot_reward(0, x, u, entropy_before)
    print(f"Robot reward with initial entropy: {robot_reward_with_info.item()}")
    
    robot_reward_with_info = robot_reward(1, x, u, entropy_after)
    print(f"Robot reward with entropy reduction: {robot_reward_with_info.item()}")