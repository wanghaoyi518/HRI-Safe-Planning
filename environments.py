import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from typing import List, Dict, Tuple, Optional, Union, Callable
import math

# Import our custom modules
from dynamics import CarDynamics
from Agents.agent import Agent
from Agents.human import HumanAgent
from Agents.robot import RobotAgent
from Agents.robot_simple import RobotSimple
from rewards import create_robot_reward, create_attentive_reward, create_distracted_reward


class Environment:
    """
    Base class for all environments.
    Handles agent registration, state updates, and visualization.
    """
    
    def __init__(self, name: str = "env"):
        """
        Initialize environment.
        
        Args:
            name: Environment identifier
        """
        self.name = name
        self.agents = {}
        self.obstacles = []
        self.time = 0
        self.collision_threshold = 0.15  # Minimum distance before collision
        self.history = []
        
    def register_agent(self, agent: Agent, agent_id: Optional[str] = None):
        """
        Register an agent in the environment.
        
        Args:
            agent: Agent to register
            agent_id: Identifier for the agent (default: agent.name)
        """
        if agent_id is None:
            agent_id = agent.name
            
        self.agents[agent_id] = agent
        
    def step(self):
        """
        Advance the environment by one time step.
        
        Returns:
            Dictionary with updated environment state
        """
        # Compute control for each agent
        controls = {}
        for agent_id, agent in self.agents.items():
            # Create environment state with information about other agents
            env_state = {
                other_id: other_agent 
                for other_id, other_agent in self.agents.items()
                if other_id != agent_id
            }
            
            # Compute control
            control = agent.compute_control(self.time, agent.state, env_state)
            controls[agent_id] = control
            
        # Apply control to each agent
        for agent_id, agent in self.agents.items():
            agent.step(controls[agent_id])
            
        # Save current state to history
        self.save_state()
        
        # Update environment time
        self.time += 1
        
        # Check for collisions and other environment-specific conditions
        info = self.get_info()
        
        return info
    
    def save_state(self):
        """Save current state to history."""
        state = {
            'time': self.time,
            'agents': {
                agent_id: {
                    'state': agent.state.clone(),
                    'action': agent.action.clone() if hasattr(agent, 'action') else None
                }
                for agent_id, agent in self.agents.items()
            }
        }
        self.history.append(state)
        
    def get_info(self) -> Dict:
        """
        Get current environment information.
        
        Returns:
            Dictionary with environment information
        """
        # Check for collisions
        collisions = self.check_collisions()
        
        return {
            'time': self.time,
            'collisions': collisions,
            'complete': self.is_complete()
        }
        
    def check_collisions(self) -> Dict[str, List[str]]:
        """
        Check for collisions between agents.
        
        Returns:
            Dictionary mapping agent IDs to lists of IDs they are colliding with
        """
        collisions = {agent_id: [] for agent_id in self.agents}
        
        # Check pairwise collisions
        for id1 in self.agents:
            for id2 in self.agents:
                if id1 != id2:
                    agent1 = self.agents[id1]
                    agent2 = self.agents[id2]
                    
                    # Compute distance between agents
                    pos1 = agent1.position
                    pos2 = agent2.position
                    dist = torch.norm(pos1 - pos2)
                    
                    if dist < self.collision_threshold:
                        collisions[id1].append(id2)
                        
        return collisions
    
    def is_complete(self) -> bool:
        """
        Check if environment has completed.
        
        Returns:
            True if environment has completed
        """
        # Base implementation: never complete
        return False
    
    def reset(self):
        """Reset environment to initial state."""
        self.time = 0
        self.history = []
        
        # Reset all agents
        for agent in self.agents.values():
            agent.reset()
            
        # Save initial state
        self.save_state()
        
    def render(self, ax=None):
        """
        Render environment and agents.
        
        Args:
            ax: Matplotlib axis (creates new figure if None)
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        # Render environment-specific elements (to be implemented by subclasses)
        self.render_environment(ax)
        
        # Render agents
        for agent_id, agent in self.agents.items():
            self.render_agent(agent, agent_id, ax)
            
        return ax
    
    def render_environment(self, ax):
        """
        Render environment-specific elements.
        To be implemented by subclasses.
        
        Args:
            ax: Matplotlib axis
        """
        pass
    
    def render_agent(self, agent: Agent, agent_id: str, ax):
        """
        Render agent.
        
        Args:
            agent: Agent to render
            agent_id: Agent identifier
            ax: Matplotlib axis
        """
        # Get agent state
        x = agent.state[0].item()
        y = agent.state[1].item()
        theta = agent.state[2].item()
        
        # Vehicle dimensions
        length = 0.2
        width = 0.1
        
        # Create rectangle for agent
        rect = Rectangle(
            (x - length/2, y - width/2),
            length, width,
            angle=math.degrees(theta),
            facecolor=agent.color,
            edgecolor='black',
            alpha=0.7,
            zorder=10
        )
        ax.add_patch(rect)
        
        # Add agent ID
        ax.text(x, y, agent_id, 
               ha='center', va='center', 
               color='white' if agent.color in ['blue', 'black'] else 'black',
               fontweight='bold',
               zorder=11)


# class Highway(Environment):
#     """
#     Highway environment with multiple lanes.
#     Implements Scenario 1: Nudging In to Explore on a Highway
#     """
    
#     def __init__(self, 
#                num_lanes: int = 2, 
#                lane_width: float = 0.13,
#                length: float = 2.0,
#                name: str = "highway"):
#         """
#         Initialize highway environment.
        
#         Args:
#             num_lanes: Number of lanes
#             lane_width: Width of each lane
#             length: Length of highway
#             name: Environment identifier
#         """
#         super().__init__(name)
        
#         self.num_lanes = num_lanes
#         self.lane_width = lane_width
#         self.length = length
        
#         # Define lane centers
#         self.lane_centers = [(i - (num_lanes-1)/2) * lane_width for i in range(num_lanes)]
        
#         # Define environment boundaries
#         self.left_boundary = self.lane_centers[0] - lane_width/2
#         self.right_boundary = self.lane_centers[-1] + lane_width/2
        
#     def get_info(self) -> Dict:
#         """
#         Get current environment information.
        
#         Returns:
#             Dictionary with environment information
#         """
#         info = super().get_info()
        
#         # Add lane information for each agent
#         info['lanes'] = {
#             agent_id: self.get_agent_lane(agent)
#             for agent_id, agent in self.agents.items()
#         }
        
#         # Add distances between agents
#         info['distances'] = {}
#         for id1 in self.agents:
#             for id2 in self.agents:
#                 if id1 != id2:
#                     agent1 = self.agents[id1]
#                     agent2 = self.agents[id2]
                    
#                     # Compute distance between agents
#                     pos1 = agent1.position
#                     pos2 = agent2.position
#                     dist = torch.norm(pos1 - pos2)
                    
#                     info['distances'][(id1, id2)] = dist.item()
        
#         return info
        
#     def get_agent_lane(self, agent: Agent) -> Optional[int]:
#         """
#         Get lane index for agent.
        
#         Args:
#             agent: Agent to get lane for
            
#         Returns:
#             Lane index or None if not in a lane
#         """
#         x = agent.position[0].item()
        
#         # Check if agent is within lane boundaries
#         if x < self.left_boundary or x > self.right_boundary:
#             return None
            
#         # Find closest lane center
#         distances = [abs(x - center) for center in self.lane_centers]
#         lane_idx = distances.index(min(distances))
        
#         return lane_idx
    
#     def render_environment(self, ax):
#         """
#         Render highway environment.
        
#         Args:
#             ax: Matplotlib axis
#         """
#         # Set axis limits
#         y_min = -self.length/2
#         y_max = self.length/2
#         x_min = self.left_boundary - self.lane_width/2
#         x_max = self.right_boundary + self.lane_width/2
        
#         ax.set_xlim(x_min, x_max)
#         ax.set_ylim(y_min, y_max)
        
#         # Draw road surface
#         road = Rectangle(
#             (self.left_boundary, y_min),
#             self.right_boundary - self.left_boundary,
#             y_max - y_min,
#             facecolor='gray',
#             alpha=0.3,
#             zorder=1
#         )
#         ax.add_patch(road)
        
#         # Draw lane markers
#         for i in range(1, self.num_lanes):
#             lane_x = self.lane_centers[i-1] + self.lane_width/2
#             ax.plot([lane_x, lane_x], [y_min, y_max], 
#                    color='white', linestyle='--', alpha=0.7, zorder=2)
            
#         # Draw road boundaries
#         ax.plot([self.left_boundary, self.left_boundary], [y_min, y_max], 
#                color='white', linestyle='-', alpha=1.0, linewidth=2, zorder=2)
#         ax.plot([self.right_boundary, self.right_boundary], [y_min, y_max], 
#                color='white', linestyle='-', alpha=1.0, linewidth=2, zorder=2)
               
#         # Add labels
#         ax.set_title(f"Highway Environment - Time: {self.time}")
        
#     def is_complete(self) -> bool:
#         """
#         Check if highway scenario has completed.
        
#         Returns:
#             True if all agents have reached the end of the highway
#         """
#         # Check if any agent is still on the highway
#         for agent in self.agents.values():
#             y = agent.position[1].item()
#             if -self.length/2 <= y <= self.length/2:
#                 return False
                
#         return True


class Intersection(Environment):
    """
    Intersection environment with four approaches.
    Implements Scenario 3: Nudging In to Explore at an Intersection
    """
    
    def __init__(self, 
                 road_width: float = 0.2,
                 intersection_size: float = 0.3,
                 name: str = "intersection"):
        """
        Initialize intersection environment.
        
        Args:
            road_width: Width of each road
            intersection_size: Size of the intersection box
            name: Environment identifier
        """
        super().__init__(name)
        
        self.road_width = road_width
        self.intersection_size = intersection_size
        
        # Define intersection boundaries
        half_size = intersection_size / 2
        self.intersection_bounds = (-half_size, half_size, -half_size, half_size)
        
        # Define stop lines (distance from center)
        self.stop_line_dist = half_size
        
        # Track which agents have crossed the intersection
        self.crossed = set()
        
    def get_info(self) -> Dict:
        """
        Get current environment information.
        
        Returns:
            Dictionary with environment information
        """
        info = super().get_info()
        
        # Add intersection information
        info['in_intersection'] = {
            agent_id: self.is_in_intersection(agent)
            for agent_id, agent in self.agents.items()
        }
        
        # Check if agents have crossed the intersection
        for agent_id, agent in self.agents.items():
            if agent_id not in self.crossed and self.has_crossed_intersection(agent):
                self.crossed.add(agent_id)
                
        # Add crossing order information
        info['crossing_order'] = list(self.crossed)
        
        return info
        
    def is_in_intersection(self, agent: Agent) -> bool:
        """
        Check if agent is in the intersection.
        
        Args:
            agent: Agent to check
            
        Returns:
            True if agent is in the intersection
        """
        x = agent.position[0].item()
        y = agent.position[1].item()
        
        x_min, x_max, y_min, y_max = self.intersection_bounds
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def has_crossed_intersection(self, agent: Agent) -> bool:
        """
        Check if agent has crossed the intersection.
        
        Args:
            agent: Agent to check
            
        Returns:
            True if agent has crossed the intersection
        """
        x = agent.position[0].item()
        y = agent.position[1].item()
        theta = agent.heading
        
        # Determine which direction the agent is heading
        heading_x = math.cos(theta)
        heading_y = math.sin(theta)
        
        # Check if agent has crossed to the other side of the intersection
        if abs(heading_x) > abs(heading_y):
            # Primarily moving in x direction
            if heading_x > 0 and x > self.intersection_bounds[1]:
                return True
            elif heading_x < 0 and x < self.intersection_bounds[0]:
                return True
        else:
            # Primarily moving in y direction
            if heading_y > 0 and y > self.intersection_bounds[3]:
                return True
            elif heading_y < 0 and y < self.intersection_bounds[2]:
                return True
                
        return False
    
    def is_on_road(self, agent: Agent) -> bool:
        """
        Check if agent is on a valid road segment.
        
        Args:
            agent: Agent to check
            
        Returns:
            True if agent is on a valid road
        """
        x = agent.position[0].item()
        y = agent.position[1].item()
        
        x_min, x_max, y_min, y_max = self.intersection_bounds
        
        # Check if in intersection box
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
            
        # Check if on horizontal road
        if abs(y) <= self.road_width/2 and -2.0 <= x <= 2.0:
            return True
            
        # Check if on vertical road
        if abs(x) <= self.road_width/2 and -2.0 <= y <= 2.0:
            return True
            
        return False
    
    def step(self):
        """
        Override step method to enforce lane constraints.
        
        Returns:
            Dictionary with updated environment state
        """
        # Compute control for each agent
        controls = {}
        for agent_id, agent in self.agents.items():
            # Create environment state with information about other agents
            env_state = {
                other_id: other_agent 
                for other_id, other_agent in self.agents.items()
                if other_id != agent_id
            }
            
            # Compute control
            control = agent.compute_control(self.time, agent.state, env_state)
            controls[agent_id] = control
            
        # Apply control to each agent
        for agent_id, agent in self.agents.items():
            # Apply control and get next state
            agent.step(controls[agent_id])
            
            # Apply position constraints to keep agent within [-2, 2] x [-2, 2]
            if agent.state[0] < -2.0:
                agent.state[0] = -2.0
            elif agent.state[0] > 2.0:
                agent.state[0] = 2.0
                
            if agent.state[1] < -2.0:
                agent.state[1] = -2.0
            elif agent.state[1] > 2.0:
                agent.state[1] = 2.0
            
            # Additional check to enforce on-road constraints
            if not self.is_on_road(agent):
                # Reset to last known valid position
                if len(agent.state_history) > 1:
                    agent.state = agent.state_history[-2].clone()
                    agent.state_history[-1] = agent.state.clone()
        
        # Save current state to history
        self.save_state()
        
        # Update environment time
        self.time += 1
        
        # Check for collisions and other environment-specific conditions
        info = self.get_info()
        
        return info
    
    def render_environment(self, ax):
        """
        Render intersection environment.
        
        Args:
            ax: Matplotlib axis
        """
        # Set axis limits to match position constraints
        limit = 2.0  # Changed from self.intersection_size * 2 to match position constraints
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        # Extract intersection boundaries
        x_min, x_max, y_min, y_max = self.intersection_bounds
        
        # Draw road segments
        # Horizontal road
        h_road = Rectangle(
            (-limit, y_min), 
            2*limit, 
            self.road_width,
            facecolor='gray',
            alpha=0.3,
            zorder=1
        )
        ax.add_patch(h_road)
        
        # Vertical road
        v_road = Rectangle(
            (x_min, -limit), 
            self.road_width, 
            2*limit,
            facecolor='gray',
            alpha=0.3,
            zorder=1
        )
        ax.add_patch(v_road)
        
        # Draw intersection box
        intersection_box = Rectangle(
            (x_min, y_min), 
            x_max - x_min, 
            y_max - y_min,
            facecolor='gray',
            alpha=0.1,
            edgecolor='white',
            linestyle=':',
            zorder=2
        )
        ax.add_patch(intersection_box)
        
        # Draw stop lines
        # Left
        ax.plot([x_min, x_min], [y_min, y_max], 
               color='white', linestyle='-', linewidth=2, zorder=3)
        # Right
        ax.plot([x_max, x_max], [y_min, y_max], 
               color='white', linestyle='-', linewidth=2, zorder=3)
        # Bottom
        ax.plot([x_min, x_max], [y_min, y_min], 
               color='white', linestyle='-', linewidth=2, zorder=3)
        # Top
        ax.plot([x_min, x_max], [y_max, y_max], 
               color='white', linestyle='-', linewidth=2, zorder=3)
            
        # Add labels
        ax.set_title(f"Intersection Environment - Time: {self.time}")
        
        # Draw boundary box to show position constraints
        boundary_box = Rectangle(
            (-2.0, -2.0),
            4.0, 4.0,
            fill=False,
            edgecolor='red',
            linestyle=':',
            linewidth=1.5,
            zorder=0
        )
        ax.add_patch(boundary_box)
        
        # Add a label for the constraint
        ax.text(-1.9, -1.9, "Position Constraints", 
               color='red', fontsize=8, ha='left', va='bottom')
        
        # Draw human goal
        human_goal = (-2.0, 0.0)
        ax.plot(human_goal[0], human_goal[1], 'r*', markersize=15, label='Human Goal')
        ax.add_patch(plt.Circle(human_goal, 0.1, color='red', alpha=0.3))
    
    def reset(self):
        """Reset intersection environment."""
        super().reset()
        self.crossed = set()
        
    def is_complete(self) -> bool:
        """
        Check if intersection scenario has completed.
        
        Returns:
            True if all agents have crossed the intersection
        """
        return len(self.crossed) == len(self.agents)


# def create_highway_scenario(robot_lane: int = 1, 
#                           human_lane: int = 0, 
#                           robot_y: float = -0.5,
#                           human_y: float = -0.1) -> Tuple[Highway, RobotAgent, HumanAgent]:
#     """
#     Create a highway scenario with robot and human agents.
    
#     Args:
#         robot_lane: Initial lane for robot (0-indexed)
#         human_lane: Initial lane for human (0-indexed)
#         robot_y: Initial y position for robot
#         human_y: Initial y position for human
        
#     Returns:
#         Tuple of (environment, robot_agent, human_agent)
#     """
#     # Create dynamics model
#     dynamics = CarDynamics(dt=0.1)
    
#     # Create environment
#     env = Highway(num_lanes=2, lane_width=0.13)
    
#     # Get lane centers
#     robot_x = env.lane_centers[robot_lane]
#     human_x = env.lane_centers[human_lane]
    
#     # Create robot agent
#     robot_init_state = torch.tensor([robot_x, robot_y, np.pi/2, 0.3])
#     robot = RobotAgent(
#         dynamics, 
#         robot_init_state,
#         reward=create_robot_reward(info_gain_weight=1.0),  # Add reward function
#         name="robot",
#         color="yellow"
#     )
    
#     # Create human agent
#     human_init_state = torch.tensor([human_x, human_y, np.pi/2, 0.5])
#     human = HumanAgent(
#         dynamics,
#         human_init_state,
#         internal_state=torch.tensor([0.8, 0.5]),  # Default: attentive
#         name="human",
#         color="red"
#     )
    
#     # Register agents
#     env.register_agent(robot)
#     env.register_agent(human)
    
#     # Register human with robot
#     robot.register_human(human)
    
#     return env, robot, human


def create_intersection_scenario(robot_direction: str = "south", 
                               human_direction: str = "west") -> Tuple[Intersection, RobotAgent, HumanAgent]:
    """
    Create an intersection scenario with robot and human agents.
    
    Args:
        robot_direction: Initial direction for robot ("north", "south", "east", "west")
        human_direction: Initial direction for human ("north", "south", "east", "west")
        
    Returns:
        Tuple of (environment, robot_agent, human_agent)
    """
    # Create dynamics model
    dynamics = CarDynamics(dt=0.1)
    
    # Create environment
    env = Intersection(road_width=0.2, intersection_size=0.3)
    
    # Get stop line positions and headings
    directions = {
        "north": (0.0, -0.5, np.pi/2),    # (x, y, theta)
        "south": (0.0, 0.5, -np.pi/2),
        "east": (-0.5, 0.0, 0.0),
        "west": (0.5, 0.0, np.pi)
    }
    
    # Create robot agent
    robot_x, robot_y, robot_theta = directions[robot_direction]
    robot_init_state = torch.tensor([robot_x, robot_y, robot_theta, 0.0])  # Initially stopped
    robot = RobotAgent(
        dynamics, 
        robot_init_state,
        reward=create_robot_reward(info_gain_weight=1.0),  # Add reward function
        name="robot",
        color="yellow"
    )
    
    # Create human agent
    human_x, human_y, human_theta = directions[human_direction]
    human_init_state = torch.tensor([human_x, human_y, human_theta, 0.0])  # Initially stopped
    human = HumanAgent(
        dynamics,
        human_init_state,
        internal_state=torch.tensor([0.8, 0.5]),  # Default: attentive
        name="human",
        color="red"
    )
    
    # Register agents
    env.register_agent(robot)
    env.register_agent(human)
    
    # Register human with robot
    robot.register_human(human)
    
    return env, robot, human

def create_intersection_scenario_simple(robot_direction: str = "south", 
                                      human_direction: str = "west") -> Tuple[Intersection, RobotSimple, HumanAgent]:
    """
    Create an intersection scenario with a simple robot model for data generation.
    
    Args:
        robot_direction: Initial direction for robot ("north", "south", "east", "west")
        human_direction: Initial direction for human ("north", "south", "east", "west")
        
    Returns:
        Tuple of (environment, robot_agent, human_agent)
    """
    # Create dynamics model
    dynamics = CarDynamics(dt=0.1)
    
    # Create environment
    env = Intersection(road_width=0.2, intersection_size=0.3)
    
    # Get stop line positions and headings
    directions = {
        "north": (0.0, -0.5, np.pi/2),    # (x, y, theta)
        "south": (0.0, 0.5, -np.pi/2),
        "east": (-0.5, 0.0, 0.0),
        "west": (0.5, 0.0, np.pi)
    }
    
    # Create simple robot agent without information gathering
    robot_x, robot_y, robot_theta = directions[robot_direction]
    robot_init_state = torch.tensor([robot_x, robot_y, robot_theta, 0.0])  # Initially stopped
    
    # Create a basic reward for the robot focused only on collision avoidance and goal reaching
    goal_position = torch.tensor([0.0, -2.0])  # Default goal at bottom of screen
    
    # Use the RobotSimple class instead of regular RobotAgent
    robot = RobotSimple(
        dynamics, 
        robot_init_state,
        reward=create_robot_reward(
            collision_weight=50.0,
            goal_weight=10.0,
            info_gain_weight=0.0,  # Zero info gain weight - crucial difference
            goal_position=goal_position
        ),
        name="robot",
        color="yellow",
        goal_position=goal_position
    )
    
    # Create human agent (same as original)
    human_x, human_y, human_theta = directions[human_direction]
    human_init_state = torch.tensor([human_x, human_y, human_theta, 0.0])  # Initially stopped
    human = HumanAgent(
        dynamics,
        human_init_state,
        internal_state=torch.tensor([0.8, 0.5]),  # Default will be overridden
        name="human",
        color="red"
    )
    
    # Register agents
    env.register_agent(robot)
    env.register_agent(human)
    
    return env, robot, human

if __name__ == "__main__":
    # Test highway environment
    # print("Creating highway scenario...")
    # highway_env, highway_robot, highway_human = create_highway_scenario()
    
    # # Render initial state
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # highway_env.render(ax1)
    
    # # Step a few times
    # for _ in range(10):
    #     highway_env.step()
    
    # # Render final state
    # highway_env.render(ax2)
    # ax1.set_title("Highway - Initial State")
    # ax2.set_title("Highway - After 10 Steps")
    
    # Test intersection environment
    print("\nCreating intersection scenario...")
    intersection_env, intersection_robot, intersection_human = create_intersection_scenario()
    
    # Render initial state
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
    intersection_env.render(ax3)
    
    # Step a few times
    for _ in range(10):
        intersection_env.step()
    
    # Render final state
    intersection_env.render(ax4)
    ax3.set_title("Intersection - Initial State")
    ax4.set_title("Intersection - After 10 Steps")
    
    plt.tight_layout()
    plt.show()