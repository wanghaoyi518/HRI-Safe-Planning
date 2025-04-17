import torch
import numpy as np
from typing import List, Dict, Callable, Tuple, Union, Optional
from features import *


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
            # Attentive drivers prioritize collision avoidance but not excessively
            # Reduced scaling effect of attentiveness (was 0.5 + 0.5 * attentiveness)
            att_factor = 0.7 + 0.3 * attentiveness
            style_factor = 1.0 - 0.5 * driving_style
            return base_weight * att_factor * style_factor
            
        elif "speed" in feature_name:
            # Aggressive drivers value speed more
            style_factor = 0.5 + 0.5 * driving_style
            return base_weight * style_factor
            
        elif "lane" in feature_name:
            # Attentive drivers stay in lanes better
            # Increased lane-following for attentive drivers (was 0.7 + 0.3)
            att_factor = 0.6 + 0.4 * attentiveness
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
            (speed_preference_feature(target_speed=1.0), 10.0),                    # Prefer driving at target speed
            (lane_following_feature(0.0), 2.0),                # Stay in lane - increased weight from 1.0 to 2.0
            (control_smoothness_feature(), 5.0),      # Smooth control inputs
            (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 50.0)      # Avoid road boundaries
        ]
        
        super().__init__(base_features=base_features, name="human_reward")


class RobotReward(Reward):
    """
    Reward function for robot planning.
    Includes components for information gain, collision avoidance, and goal-directed behavior.
    """
    
    def __init__(self, 
                 collision_avoidance_features: List[Tuple[Feature, float]] = None,
                 goal_features: List[Tuple[Feature, float]] = None,
                 info_gain_weight: float = 1.0):
        """
        Initialize robot reward with different reward components.
        
        Args:
            collision_avoidance_features: Features for collision avoidance
            goal_features: Features for goal-directed behavior
            info_gain_weight: Weight for information gain component
        """
        # Initialize with basic features if none provided
        collision_features = collision_avoidance_features or []
        goal_features = goal_features or []
        
        # Combine all features into a single list
        all_features = collision_features + goal_features
        
        super().__init__(features=all_features, name="robot_reward")
        
        self.collision_avoidance_features = collision_features
        self.goal_features = goal_features
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
            Total reward combining collision avoidance, goal, and information gain
        """
        # Compute task reward (collision avoidance + goal)
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
    
    def get_collision_avoidance_reward(self, t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute just the collision avoidance component of the reward.
        
        Args:
            t: Time step
            x: State vector
            u: Control vector
            
        Returns:
            Collision avoidance reward
        """
        reward = torch.tensor(0.0, dtype=torch.float32)
        
        for feature, weight in self.collision_avoidance_features:
            feature_value = feature(t, x, u)
            reward = reward + weight * feature_value
            
        return reward
    
    def get_goal_reward(self, t: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute just the goal-directed component of the reward.
        
        Args:
            t: Time step
            x: State vector
            u: Control vector
            
        Returns:
            Goal-directed reward
        """
        reward = torch.tensor(0.0, dtype=torch.float32)
        
        for feature, weight in self.goal_features:
            feature_value = feature(t, x, u)
            reward = reward + weight * feature_value
            
        return reward



# def create_parameterized_human_reward(internal_state: torch.Tensor) -> HumanReward:
#     """
#     Create a human reward function parameterized by internal state.
    
#     Args:
#         internal_state: Human internal state [attentiveness, driving_style]
        
#     Returns:
#         Personalized HumanReward based on internal state
#     """
#     # Extract attentiveness and driving style
#     attentiveness = internal_state[0].item()
#     driving_style = internal_state[1].item()
    
#     # Set human goal
#     human_goal = torch.tensor([-2.0, 0.0])
    
#     # Create base reward
#     reward = HumanReward()
    
#     # Calculate target speed based on driving style (aggressive = faster)
#     # Range from 0.6 (most conservative) to 1.8 (most aggressive)
#     target_speed = 0.6 + 1.2 * driving_style
    
#     # Calculate speed weight - aggressive drivers prioritize speed more
#     # Range from 7.0 to 17.0
#     speed_weight = 7.0 + 10.0 * driving_style
    
#     # Calculate lane following weight - attentive drivers follow lanes better
#     # Range from 0.5 to 4.5
#     lane_weight = 0.5 + 4.0 * attentiveness
    
#     # Calculate control smoothness weight - conservative drivers have smoother controls
#     # Range from 2.0 to 8.0 with inverse relationship to driving style
#     control_weight = 8.0 - 6.0 * driving_style
    
#     # Calculate boundary awareness - highly affected by both attentiveness and driving style
#     # Range from 20.0 to 80.0
#     boundary_weight = 20.0 + 60.0 * attentiveness * (1.0 - 0.5 * driving_style)
    
#     # Calculate goal weight - attentive and aggressive drivers are more goal-focused
#     # Range from 3.0 to 15.0
#     goal_weight = 2*(3.0 + 6.0 * attentiveness + 6.0 * driving_style)
    
#     # Create the parameterized feature set with calculated weights
#     reward.base_features = [
#         (speed_preference_feature(target_speed=target_speed), speed_weight),
#         (lane_following_feature(0.0), lane_weight),
#         (control_smoothness_feature(), control_weight),
#         (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), boundary_weight),
#         (create_goal_reaching_feature(human_goal, weight=goal_weight), 1.0)
#     ]
    
#     # Add special behaviors for extreme internal states
    
#     # Very distracted behavior (att <= 0.2)
#     if attentiveness <= 0.2:
#         # Add random decision feature to simulate unpredictable behavior
#         def random_decision(t, x, u):
#             # Seed based on time and position for consistency
#             seed = int(t * 10 + x[0].item() * 100) % 1000
#             torch.manual_seed(seed)
#             return torch.randn(1).item()
        
#         distraction_feature = Feature(random_decision)
#         reward.base_features.append((distraction_feature, 3.0))
    
#     # Very aggressive behavior (style > 0.9)
#     if driving_style >= 0.8:
#         # Add impatience feature - penalty proportional to time spent in same position
#         def impatience_feature(t, x, u):
#             # Penalize low speeds more heavily as time increases
#             return -0.1 * t * torch.abs(x[3] - target_speed)
        
#         reward.base_features.append((Feature(impatience_feature), 2.0))
    
#     # Very careful behavior (att > 0.9 and style < 0.3)
#     if attentiveness >= 0.8 and driving_style < 0.3:
#         # Add extra caution near boundaries
#         boundary_buffer = 0.05
#         def extra_caution_feature(t, x, u):
#             # Add penalty for being close to lane edges or other boundaries
#             pos = x[:2]
#             dist_to_center = torch.abs(pos[0])  # Distance from lane center
#             return -5.0 * torch.exp(-10.0 * (dist_to_center - 0.1)**2)
        
#         reward.base_features.append((Feature(extra_caution_feature), 5.0))
    
#     return reward

def create_parameterized_human_reward(internal_state: torch.Tensor) -> HumanReward:
    """
    Create a human reward function parameterized by internal state with highly differentiated behaviors
    and significant emphasis on goal reaching.
    
    Args:
        internal_state: Human internal state [attentiveness, driving_style]
        
    Returns:
        Personalized HumanReward based on internal state
    """
    # Extract attentiveness and driving style
    attentiveness = internal_state[0].item()
    driving_style = internal_state[1].item()
    
    # Set human goal
    human_goal = torch.tensor([-2.0, 0.0])
    
    # Create base reward
    reward = HumanReward()
    
    # Handle each specific combination of internal states
    
    # 1. EXTREMELY DISTRACTED & VERY CAUTIOUS (0.2, 0.2)
    if attentiveness == 0.2 and driving_style == 0.2:
        reward.base_features = [
            (speed_preference_feature(target_speed=0.5), 6.0),         # Very slow speed
            (lane_following_feature(0.0), 0.3),                        # Terrible lane following due to distraction
            (control_smoothness_feature(), 8.0),                       # Very smooth but ineffective controls
            (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 15.0),    # Minimal boundary awareness
            (create_goal_reaching_feature(human_goal, weight=7.0), 4.0) # Significant goal focus despite low attention
        ]
        
        
    
    # 2. DISTRACTED & MODERATE STYLE (0.2, 0.5)
    elif attentiveness == 0.2 and driving_style == 0.5:
        reward.base_features = [
            (speed_preference_feature(target_speed=1.1), 10.0),        # Moderate+ speed despite distraction
            (lane_following_feature(0.0), 0.4),                        # Poor lane following
            (control_smoothness_feature(), 3.0),                       # Jerky controls
            (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 12.0),    # Very low boundary awareness
            (create_goal_reaching_feature(human_goal, weight=8.0), 5.0) # Enhanced goal focus
        ]
        
    
    # 3. DISTRACTED & AGGRESSIVE (0.2, 0.8)
    elif attentiveness == 0.2 and driving_style == 0.8:
        reward.base_features = [
            (speed_preference_feature(target_speed=1.8), 25.0),        # Extremely high speed
            (lane_following_feature(0.0), 0.2),                        # Almost no lane discipline
            (control_smoothness_feature(), 1.0),                       # Minimal control smoothness
            (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 4.0),     # Dangerously low boundary awareness
            (create_goal_reaching_feature(human_goal, weight=15.0), 20.0) # Strong goal focus despite distraction
        ]
        
    
    # 4. MODERATE ATTENTION & CONSERVATIVE (0.5, 0.2)
    elif attentiveness == 0.5 and driving_style == 0.2:
        reward.base_features = [
            (speed_preference_feature(target_speed=0.6), 15.0),        # Very consistent slow speed
            (lane_following_feature(0.0), 3.5),                        # Good lane following
            (control_smoothness_feature(), 9.0),                       # Extremely smooth controls
            (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 60.0),    # High boundary awareness
            (create_goal_reaching_feature(human_goal, weight=10.0), 12.0) # Boosted goal focus
        ]
                
    # Focus on these combinations specifically
    elif attentiveness == 0.5 and driving_style == 0.5:  # MODERATE ATTENTION & MODERATE STYLE (0.5, 0.5)
        reward.base_features = [
            (speed_preference_feature(target_speed=0.9), 10.0),        # Slightly slower than average
            (lane_following_feature(0.0), 2.0),                        # Standard lane following
            (control_smoothness_feature(), 5.0),                       # Medium control smoothness
            (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 40.0),    # Standard boundary awareness
            (create_goal_reaching_feature(human_goal, weight=10.0), 5.0) # Moderate goal focus
        ]
        
        # Add mild acceleration preference - very standard driving
        def standard_driving(t, x, u):
            current_speed = x[3]
            target_speed = 0.9
            
            # Prefer gentle acceleration and deceleration
            return -3.0 * (current_speed - target_speed)**2 - 1.0 * torch.sum(u**2)
        
        reward.base_features.append((Feature(standard_driving), 3.0))

    elif attentiveness == 0.5 and driving_style == 0.8:  # MODERATE ATTENTION & AGGRESSIVE (0.5, 0.8)
        reward.base_features = [
            (speed_preference_feature(target_speed=1.8), 25.0),        # MUCH higher speed
            (lane_following_feature(0.0), 0.5),                        # Very minimal lane following
            (control_smoothness_feature(), 1.0),                       # Minimal control smoothness
            (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 15.0),    # Very low boundary awareness
            (create_goal_reaching_feature(human_goal, weight=25.0), 15.0) # Extremely high goal focus
        ]
        
        # Add extreme impatience
        def extreme_impatience(t, x, u):
            current_speed = x[3]
            
            # Extremely reward high speeds and acceleration
            return 8.0 * current_speed + 10.0 * torch.max(u[1], torch.tensor(0.0))
        
        # Add aggressive lane changing
        def aggressive_lane_changes(t, x, u):
            # Reward dramatic steering inputs
            return -2.0 * (u[0]**2 - 0.4)**2  # Prefer ~0.6 steering
        
        # Add direct line to goal behavior
        def direct_line_to_goal(t, x, u):
            # Calculate vector to goal
            pos = x[:2]
            goal_dir = (human_goal - pos) / torch.norm(human_goal - pos)
            heading_vec = torch.tensor([torch.cos(x[2]), torch.sin(x[2])])
            
            # Reward alignment with direct path to goal
            alignment = torch.dot(goal_dir, heading_vec)
            return 5.0 * alignment
        
        reward.base_features.append((Feature(extreme_impatience), 7.0))
        reward.base_features.append((Feature(aggressive_lane_changes), 5.0))
        reward.base_features.append((Feature(direct_line_to_goal), 6.0))

    elif attentiveness == 0.8 and driving_style == 0.5:  # HIGHLY ATTENTIVE & MODERATE STYLE (0.8, 0.5)
        reward.base_features = [
            (speed_preference_feature(target_speed=1.2), 15.0),        # Efficient moderate speed
            (lane_following_feature(0.0), 6.0),                        # Exceptional lane following
            (control_smoothness_feature(), 8.0),                       # Very smooth controls
            (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 85.0),    # Very high boundary awareness
            (create_goal_reaching_feature(human_goal, weight=18.0), 10.0) # Strong goal focus with precision
        ]
        
        # Add perfect trajectory optimization
        def perfect_trajectory(t, x, u):
            # Simulate perfect planning and execution
            pos = x[:2]
            dist_to_goal = torch.norm(human_goal - pos)
            
            # Calculate ideal trajectory with gradual deceleration
            ideal_deceleration_start = 0.8  # Start slowing at this distance
            
            if dist_to_goal < ideal_deceleration_start:
                ideal_speed = max(0.3, dist_to_goal / ideal_deceleration_start * 1.2)
                return -8.0 * (x[3] - ideal_speed)**2
            
            return torch.tensor(0.0)
        
        # Add anticipatory driving
        def anticipatory_control(t, x, u):
            # Reward smooth, gradual control changes
            return -6.0 * torch.sum(u**2) + 4.0 * x[3]  # Prefer smooth, efficient motion
        
        reward.base_features.append((Feature(perfect_trajectory), 7.0))
        reward.base_features.append((Feature(anticipatory_control), 5.0))

    elif attentiveness == 0.8 and driving_style == 0.8:  # HIGHLY ATTENTIVE & AGGRESSIVE (0.8, 0.8)
        reward.base_features = [
            (speed_preference_feature(target_speed=5.0), 30.0),        # Maximum speed possible
            (lane_following_feature(0.0), 1.5),                        # Minimal but precise lane adherence
            (control_smoothness_feature(), 2.0),                       # Low smoothness - sharp maneuvers
            (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), 30.0),    # Moderate boundary awareness
            (create_goal_reaching_feature(human_goal, weight=30.0), 20.0) # Maximum goal focus possible
        ]
        
        # Add extreme racing behavior
        def extreme_racing(t, x, u):
            current_speed = x[3]
            # Massively reward speed, with exponential scaling
            return 10.0 * current_speed**2
        
        # Add "cut the corner" behavior
        def cut_corner(t, x, u):
            # Rewards taking shortcuts directly to goal
            pos = x[:2]
            dist_to_goal = torch.norm(human_goal - pos)
            
            # Reward direct paths even if they cut through intersection
            return -15.0 * dist_to_goal
        
        # Add dramatic acceleration and braking
        def dramatic_driving(t, x, u):
            # Reward extreme inputs
            return 8.0 * torch.abs(u[1]) - 2.0 * (u[1]**2 - 0.8)**2  # Prefers ~0.9 acceleration
        
        reward.base_features.append((Feature(extreme_racing), 10.0))
        reward.base_features.append((Feature(cut_corner), 8.0))
        reward.base_features.append((Feature(dramatic_driving), 6.0))
        
        # Default case - use a formula-based approach for any other values
    else:
        print(f"formula-based with attentiveness: {attentiveness:.2f}, driving style: {driving_style:.2f}")
        # Calculate target speed based on driving style (aggressive = faster)
        # Expanded range from 0.5 (most conservative) to 2.0 (most aggressive)
        target_speed = 0.5 + 1.5 * driving_style
        
        # Calculate speed weight - aggressive drivers prioritize speed much more
        # Expanded range from 5.0 to 30.0
        speed_weight = 5.0 + 25.0 * driving_style
        
        # Calculate lane following weight - attentive drivers follow lanes much better
        # Expanded range from 0.2 to 6.0 with stronger attentiveness effect
        lane_weight = 0.2 + 5.8 * attentiveness
        
        # Calculate control smoothness weight - conservative drivers have smoother controls
        # Expanded range from 1.0 to 12.0 with inverse relationship to driving style
        control_weight = 12.0 - 11.0 * driving_style
        
        # Calculate boundary awareness - highly affected by both attentiveness and driving style
        # Expanded range from 5.0 to 100.0
        boundary_weight = 5.0 + 95.0 * attentiveness * (1.0 - 0.7 * driving_style)
        
        # Calculate goal weight - significantly increased for all types
        # Expanded range from 5.0 to 30.0 (increased base level)
        goal_weight = 5.0 + 10.0 * attentiveness + 15.0 * driving_style
        
        # Create the parameterized feature set with calculated weights
        reward.base_features = [
            (speed_preference_feature(target_speed=target_speed), speed_weight),
            (lane_following_feature(0.0), lane_weight),
            (control_smoothness_feature(), control_weight),
            (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), boundary_weight),
            (create_goal_reaching_feature(human_goal, weight=goal_weight), 5.0)  # Significant base multiplier
        ]
        
        # Add special behaviors for extreme values
        
        # Very distracted behavior (att < 0.3)
        if attentiveness < 0.3:
            def distraction_effects(t, x, u):
                # Seed based on time for consistency
                seed = int(t * 7) % 1000
                torch.manual_seed(seed)
                random_factor = torch.rand(1).item()
                
                # Occasional complete inattention (10% chance)
                if random_factor < 0.1:
                    return -5.0 * torch.sum(u**2)  # Reward no control input
                
                # Other times, erratic control
                return 2.0 * torch.randn(1).item() * u[0]
            
            reward.base_features.append((Feature(distraction_effects), 3.0))
        
        # Very aggressive behavior (style > 0.7)
        if driving_style > 0.7:
            def aggression_effects(t, x, u):
                current_speed = x[3]
                # Strong reward for acceleration and high speeds
                return 3.0 * u[1] + 2.0 * current_speed
            
            reward.base_features.append((Feature(aggression_effects), 4.0 * driving_style))
    
    return reward


def create_robot_reward(collision_weight: float = 50.0, 
                       goal_weight: float = 10.0,
                       info_gain_weight: float = 1.0,
                       goal_position=None,
                       obstacle_positions=None) -> RobotReward:
    """
    Create a comprehensive reward function for the robot.
    
    Args:
        collision_weight: Weight for collision avoidance features
        goal_weight: Weight for goal-directed features
        info_gain_weight: Weight for information gain component
        goal_position: Target position [x, y]
        obstacle_positions: List of obstacle positions
        
    Returns:
        Robot reward function
    """
    # Default positions if none provided
    if goal_position is None:
        goal_position = torch.tensor([0, -2.0])
    
    if obstacle_positions is None:
        obstacle_positions = []
    
    # Define collision avoidance features
    collision_features = [
        (road_boundary_feature([[1, 0, 1], [-1, 0, 1]]), collision_weight),  # Avoid road boundaries
        (robot_obstacle_avoidance_feature(obstacle_positions), collision_weight)  # Avoid obstacles
    ]
    
    # Define goal-directed features
    goal_features = [
        (speed_preference_feature(target_speed=1.2), goal_weight * 0.7),  # Increased from 0.8 to 1.2 and weight from 0.5 to 0.7
        (create_goal_reaching_feature(goal_position, goal_weight), 1.5)  # Increased from 1.0 to 1.5
    ]
    
    return RobotReward(
        collision_avoidance_features=collision_features,
        goal_features=goal_features,
        info_gain_weight=info_gain_weight
    )


if __name__ == "__main__":
    # Test reward functions
    
    # Create test state and control
    x = torch.tensor([0.0, 0.0, 0.0, 1.2])  # [x, y, theta, v]
    u = torch.tensor([0.1, 0.5])            # [steering, acceleration]
    
    # Test human reward
    human_reward = create_parameterized_human_reward(internal_state=torch.tensor([0.5, 0.5]))
    reward_value = human_reward(0, x, u)
    print(f"Normal human reward: {reward_value.item()}")
    
    # Test with different internal states
    attentive_aggressive = torch.tensor([0.9, 0.8])  # High attentiveness, high aggressiveness
    distracted_conservative = torch.tensor([0.2, 0.3])  # Low attentiveness, low aggressiveness
    
    reward_att_agg = human_reward(0, x, u, attentive_aggressive)
    reward_dis_con = human_reward(0, x, u, distracted_conservative)
    
    print(f"Reward (attentive & aggressive): {reward_att_agg.item()}")
    print(f"Reward (distracted & conservative): {reward_dis_con.item()}")
    
    # Set up a goal and obstacle for robot reward
    goal_pos = torch.tensor([5.0, 5.0])
    obstacle_pos = [torch.tensor([2.0, 2.0])]
    
    # Test robot reward with information gain and different components
    robot_reward = create_robot_reward(
        collision_weight=50.0,
        goal_weight=10.0,
        info_gain_weight=2.0,
        goal_position=goal_pos,
        obstacle_positions=obstacle_pos
    )
    
    task_reward = robot_reward(0, x, u)
    print(f"Robot task reward: {task_reward.item()}")
    print(f"- Collision avoidance component: {robot_reward.get_collision_avoidance_reward(0, x, u).item()}")
    print(f"- Goal-directed component: {robot_reward.get_goal_reward(0, x, u).item()}")
    
    # Test with information gain
    entropy_before = torch.tensor(1.5)  # Higher entropy (more uncertainty)
    entropy_after = torch.tensor(0.8)   # Lower entropy (less uncertainty)
    
    robot_reward_with_info = robot_reward(0, x, u, entropy_before)
    print(f"Robot reward with initial entropy: {robot_reward_with_info.item()}")
    
    robot_reward_with_info = robot_reward(1, x, u, entropy_after)
    print(f"Robot reward with entropy reduction: {robot_reward_with_info.item()}")