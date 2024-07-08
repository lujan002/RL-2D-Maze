import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Box2D
from Box2D.b2 import (world, vec2, polygonShape, staticBody, dynamicBody)
import pygame
from pygame.locals import QUIT
import random
from Maze import Maze
import time
import astar_pathfinding as astar

max_episode_steps = 256

# Maze generation functions
def generate_maze(width, height):
    newMaze = Maze(width,height)
    newMaze.makeMazeGrowTree(weightHigh = 10, weightLow = 0) #higher both weights are, the harder the maze is to solve
    newMaze.makeMazeBraided(-1) #removes all dead ends
    # newMaze.makeMazeBraided(7) #or introduces random loops, by taking a percentage of tiles that will have additional connections
    mazeImageBW = newMaze.makePP()
    # mazeImageBW.show() #can or can not work, see Pillow documentation. For debuging only

    return newMaze.get_grid()

# Create an empty grid with only the border walls
def generate_empty_maze(width, height):
	newMaze = Maze(width, height)
	maze_grid = newMaze.get_grid()

	# Clear internal walls and set border walls
	for row in maze_grid:
		for tile in row:
			# Clear all connections first
			tile.connectTo = []

			# Set border walls
			if tile.coordinateX == 0:
				tile.connectTo.append("W")
			if tile.coordinateX == width - 1:
				tile.connectTo.append("E")
			if tile.coordinateY == 0:
				tile.connectTo.append("N")
			if tile.coordinateY == height - 1:
				tile.connectTo.append("S")

	return maze_grid

class RayCastClosestCallback(Box2D.b2RayCastCallback):
    def __init__(self):
        # Box2D.b2RayCastCallback.__init__(self)
        super().__init__()
        self.hit = False
        self.point = Box2D.b2Vec2()
        self.normal = Box2D.b2Vec2()
        self.fraction = 1.0

    def ReportFixture(self, fixture, point, normal, fraction):
        self.hit = True
        self.point = point
        self.normal = normal
        self.fraction = fraction
        return fraction

class ContinuousMazeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, seed=None):
        super(ContinuousMazeEnv, self).__init__()
        self.render_mode = render_mode
        self.seed = seed
        
        # Define action and observation space
        self.action_space = spaces.Box(low=np.array([0, -1, -1, -1]), high=np.array([1, 1, 1, 1]), dtype=np.float32)

        # Continuous observation space: [position_x, position_y, velocity_x, velocity_y, orientation, goal_x, goal_y, lidar_array]
        # Continuous observation space: [position_relative_x, position_relative_y, orientation, collision, lidar_array]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + 1 + 1 + 8,), dtype=np.float32)

        # Pygame initialization
        self.cell_size = 40  # Each cell is _x_ pixels
        self.grid_width = 3  # _ cells wide
        self.grid_height = 3  # _ cells tall
        self.screen_width = (self.cell_size * (self.grid_width * 2  + 1) // 16) * 16
        self.screen_height = (self.cell_size * (self.grid_height * 2  + 1) // 16) * 16
        self.scale = 1  # Pixels per meter (since cell size is already in pixels)

        # Initialize the Box2D world
        self.world = world(gravity=(0, 0))

        # Create the agent
        # self.agent = self.world.CreateDynamicBody(position=(0, 0), linearDamping=0.5, angularDamping=0.5)
        self.agent = self.world.CreateDynamicBody(
            # position=(((self.cell_size/4)) / self.scale, ((self.cell_size/4)) / self.scale), 
            position = (50,50),
            linearDamping=0.0, 
            angularDamping=0.0)
        # self.agent.CreateCircleFixture(radius=10, density=1.0, friction=0.3)

        # Define the vertices of a rectangle centered at the agent's position
        self.half_width = 3
        self.half_height = 5
        self.agent_vertices = [(-self.half_width, -self.half_height), (self.half_width, -self.half_height), (self.half_width, self.half_height), (-self.half_width, self.half_height)]
        self.agent.CreatePolygonFixture(vertices=self.agent_vertices, density=1.0, friction=0.3)
        # print(f"Agent created at position: {self.agent.position}")  # Debugging line

        # setting constants for angular rotation physics
        self.agent.mass = 1
        self.agent.moment_of_inertia = self.agent.mass * (self.agent.fixtures[0].shape.radius ** 2) / 2

        self.screen = None
        self.clock = None
        self.viewer = None
        self.font = None
        #self.agent.initial_position = (50,50)
        self.agent.orientation = 0.0
        self.timesteps = 0.0
        self.max_lidar_dist = 100
        self.previous_lidar_reward = 0.0
        self.relative_position = (0.0, 0.0) # Initialization
        self.visit_count = {}
        #self.new_reward_proximity = 0.0
        self.old_reward_proximity = None
        self.truncated_count = 0.0
        self.record_distance = 9999999

        self.breadcrumbs_collected = []  # Track collected breadcrumbs
        self.reset()

    def generate_goal(self, seed):
        if seed is not None:
            random.seed(seed)
        while True:
            # Generate a random position within the maze boundaries
            goal_x = random.uniform(0, self.screen_width / self.scale)
            goal_y = random.uniform(0, self.screen_height / self.scale)

            # Calculate the distance between the agent's initial position and the goal
            distance_to_agent = np.sqrt((goal_x - self.agent.initial_position[0])**2 + (goal_y - self.agent.initial_position[1])**2)

            # Check if the goal is not too close to the agent and not in a wall
            if distance_to_agent >= 100 / self.scale and not self.is_agent_collision((goal_x, goal_y)):
                return (goal_x, goal_y)

    def is_lidar_collision(self, position):
        agent_x, agent_y = position
        # print(position)
        for row in self.maze_grid:
            for tile in row:
                x = tile.coordinateX * 2 * self.cell_size
                y = tile.coordinateY * 2 * self.cell_size

                # Check if the agent is within the bounds of this cell (even though each cell is 3x3, there is some overlap, so just take each cell as 2x2 to avoid overlap)
                if x <= agent_x < x + self.cell_size * 3 and y <= agent_y < y + self.cell_size * 3:
                    # Check for collisions with the walls
                    if "N" not in tile.connectTo and y <= agent_y < y + self.cell_size and x + self.cell_size <= agent_x < x + self.cell_size * 2:
                        # print("agent is in a forbidden N tile")
                        return True
                    if "S" not in tile.connectTo and y + self.cell_size *2 <= agent_y < y + self.cell_size * 3 and x + self.cell_size <= agent_x < x + self.cell_size * 2:
                        # print("agent is in a forbidden S tile")
                        return True
                    if "W" not in tile.connectTo and x <= agent_x < x + self.cell_size and y + self.cell_size <= agent_y < y + self.cell_size * 2:
                        # print("agent is in a forbidden W tile")
                        return True
                    if "E" not in tile.connectTo and x + self.cell_size * 2 <= agent_x < x + self.cell_size * 3 and y + self.cell_size <= agent_y < y + self.cell_size * 2:
                        # print("agent is in a forbidden E tile")
                        return True
                    if x <= agent_x < x + self.cell_size and y <= agent_y < y + self.cell_size:
                        # print("agent is in a forbidden origin (top left) tile")
                        return True
                    if x + self.cell_size * 2 <= agent_x < x + self.cell_size * 3 and y + self.cell_size * 2 <= agent_y < y + self.cell_size * 3:
                        # print("agent is in a forbidden bottom right tile")
                        return True
                    
                    # Additional checks for bottom-left and top-right corners
                    if x <= agent_x < x + self.cell_size and y + self.cell_size * 2 <= agent_y < y + self.cell_size * 3:
                        return True
                    if x + self.cell_size * 2 <= agent_x < x + self.cell_size * 3 and y <= agent_y < y + self.cell_size:
                        return True
                    
        return False
    
    def is_agent_collision(self, position):
        # Rotate and translate vertices to the global frame
        global_vertices = []
        for vertex in self.agent_vertices:
            x_local, y_local = vertex
            x_global = x_local * np.cos(self.agent.orientation) - y_local * np.sin(self.agent.orientation) + position[0]
            y_global = x_local * np.sin(self.agent.orientation) + y_local * np.cos(self.agent.orientation) + position[1]
            global_vertices.append((x_global, y_global))

        for vertex in global_vertices:
            agent_x, agent_y = vertex
            for row in self.maze_grid:
                for tile in row:
                    x = tile.coordinateX * 2 * self.cell_size
                    y = tile.coordinateY * 2 * self.cell_size

                    # Check if the agent is within the bounds of this cell (even though each cell is 3x3, there is some overlap, so just take each cell as 2x2 to avoid overlap)
                    if x <= agent_x < x + self.cell_size * 3 and y <= agent_y < y + self.cell_size * 3:
                        # Check for collisions with the walls
                        if "N" not in tile.connectTo and y <= agent_y < y + self.cell_size and x + self.cell_size <= agent_x < x + self.cell_size * 2:
                            # print("agent is in a forbidden N tile")
                            return True
                        if "S" not in tile.connectTo and y + self.cell_size *2 <= agent_y < y + self.cell_size * 3 and x + self.cell_size <= agent_x < x + self.cell_size * 2:
                            # print("agent is in a forbidden S tile")
                            return True
                        if "W" not in tile.connectTo and x <= agent_x < x + self.cell_size and y + self.cell_size <= agent_y < y + self.cell_size * 2:
                            # print("agent is in a forbidden W tile")
                            return True
                        if "E" not in tile.connectTo and x + self.cell_size * 2 <= agent_x < x + self.cell_size * 3 and y + self.cell_size <= agent_y < y + self.cell_size * 2:
                            # print("agent is in a forbidden E tile")
                            return True
                        if x <= agent_x < x + self.cell_size and y <= agent_y < y + self.cell_size:
                            # print("agent is in a forbidden origin (top left) tile")
                            return True
                        if x + self.cell_size * 2 <= agent_x < x + self.cell_size * 3 and y + self.cell_size * 2 <= agent_y < y + self.cell_size * 3:
                            # print("agent is in a forbidden bottom right tile")
                            return True
                        
                        # Additional checks for bottom-left and top-right corners
                        if x <= agent_x < x + self.cell_size and y + self.cell_size * 2 <= agent_y < y + self.cell_size * 3:
                            return True
                        if x + self.cell_size * 2 <= agent_x < x + self.cell_size * 3 and y <= agent_y < y + self.cell_size:
                            return True
        return False
    
    # def cast_ray(self, angle):
    #     start_point = vec2(self.agent.position)
    #     max_distance = 100.0
    #     step_size = 5.0  # The step size for checking along the ray
    #     direction = vec2(np.cos(angle), np.sin(angle))
        
    #     # Iterate along the ray path in small steps
    #     for step in np.arange(0, max_distance, step_size):
    #         end_point = start_point + step * direction
    #         if self.is_lidar_collision(end_point):
    #             # Return the distance to the collision point
    #             distance = np.linalg.norm(end_point - start_point)
    #             return distance

    #     if not self.is_lidar_collision(end_point):
    #         return max_distance
    
    def cast_ray(self, angle):
        start_point = vec2(self.agent.position)
        direction = vec2(np.cos(angle), np.sin(angle))

        # Binary search parameters
        left, right = 0, self.max_lidar_dist # Max distance of 100 units
        precision = 0.1

        while right - left > precision:
            mid = (left + right) / 2
            mid_point = start_point + mid * direction
            quarter_point = start_point + (mid/2) * direction
            # Ensure mid_point is within bounds
            if mid_point[0] <= 0 or mid_point[0] >= self.screen_width or mid_point[1] <= 0 or mid_point[1] >= self.screen_height:
                #print('lidar end out of bounds')
                right = mid  # If out of bounds, adjust right to narrow the search
                continue

            if self.is_lidar_collision(mid_point) or self.is_lidar_collision(quarter_point): #check quarter point for edge cases where mid point is not in a wall but quarter point is
                right = mid
            else:
                left = mid

        return (left + right) / 2

    def calc_gaussian(self, x, lambda_val):
        return np.exp(-lambda_val * x**2)
    
    def calc_gaussian_2(self, x, lambda_val):
        return 2*np.exp(-lambda_val * x**2)-1
        
    def calc_exp_decay(self, x, lambda_val):
        # lambda_value = 4 * np.log(2) / np.pi**2 # Lambda for with gaussian dist (x^2)
        relative_orientation = np.exp(-lambda_val * abs(x))

        # print(f"desired_orientation: {desired_orientation}")
        # print(f"agent_orientation: {agent_orientation}")
        return relative_orientation
    
    def step(self, action):
        # print(f"Received action: {action}")
        self.timesteps += 1
        # print(f"Timestep: {self.timesteps}")

        # Store the old position for potential collision checks
        old_position = np.array(self.agent.position)
        old_orientation = self.agent.orientation
        
        # Process actions
        discrete_action = int(round(action[0]))  # Discrete action (0 or 1)
        translation_action = action[1:3]
        rotation_action = action[3]
        if discrete_action == 0:
            # Directly update the position based on the action
            side_velocity = translation_action[0] * 2.5  # Adjust scaling factor as needed
            forward_velocity = translation_action[1] * 10  # Adjust scaling factor as needed
        
            # Calculate the new position
            dx = side_velocity * np.cos(self.agent.orientation) - forward_velocity * np.sin(self.agent.orientation)
            dy = side_velocity * np.sin(self.agent.orientation) + forward_velocity * np.cos(self.agent.orientation)
            
            new_position = old_position + np.array([dx, dy])
            dtheta = 0 # No roation if moving
        else:
		    # Handle rotation
            dtheta = rotation_action * np.pi / 4  # Adjust scaling factor as needed
            dx, dy = 0, 0  # No translation if rotating
            new_position = old_position
            

        # Incrementally check for collisions
        collision_steps = 10
        incremental_translation = np.array([dx, dy]) / collision_steps       
        incremental_rotation = dtheta / collision_steps
        collision_detected = False
        for _ in range(collision_steps):
            self.agent.position += incremental_translation
            self.agent.orientation += incremental_rotation
            if self.is_agent_collision(self.agent.position):
                self.agent.position -= incremental_translation
                self.agent.orientation -= incremental_rotation
                collision_detected=True
                break
        self.agent.orientation = (self.agent.orientation + (np.pi * 2)) % (np.pi * 2)
        old_relative_position = self.relative_position
        # # Calculate the relative position to the goal using Gaussian function
        relative_x = self.calc_exp_decay(self.goal_position[0] - self.agent.position[0], 0.01)
        relative_y = self.calc_exp_decay(self.goal_position[1] - self.agent.position[1], 0.01)
        self.relative_position = (relative_x, relative_y)


        # Relative orientation 
        dx = self.goal_position[0] - self.agent.position[0]
        dy = self.goal_position[1] - self.agent.position[1]
        # Calculate the desired angle to face the goal
        desired_orientation = np.arctan2(dy, dx)
        # Adjust the desired orientation to have 0 facing south
        desired_orientation = (desired_orientation + 1.5 * np.pi) % (2 * np.pi)
        # print(f"desired orientation: {desired_orientation}")
        # Calculate the difference in orientation
        orientation_diff = self.agent.orientation - desired_orientation
        # print(f"orientation diff before: {orientation_diff}")
        # Adjust orientation_diff to that it is positive when goal is to left of agent and negative when right of agent, and 0 being facing the goal and +/-pi being back to the goal
        if orientation_diff > np.pi or orientation_diff < -np.pi:
            if orientation_diff  >= 0:
                orientation_diff = -(np.pi - (orientation_diff % (np.pi)))
            elif orientation_diff < 0:
                orientation_diff = orientation_diff + 2*np.pi #(np.pi + (orientation_diff % (np.pi)))

        # print(f"orientation diff after: {orientation_diff}")
        # relative_orientation = self.calc_exp_decay(orientation_diff, 2) # ensures that curve flattens out to zero around x=pi  
        relative_orientation = orientation_diff / np.pi # normalize with a linear function so that 0 maps to 0, π maps to 1, and -π maps to -1. Hopefully this gives better orientation vision to the agent.
        # if orientation_diff < 0:
        #     goal_side = 1 # if goal is to the right of the agent
        # else:
        #     goal_side = 0 # if goal is to the left of the agent

        # print(f"relative orientation: {relative_orientation}")
        # print(f"agent orientation: {self.agent.orientation}")


        # Perform ray casting
        num_rays = 8
        angles = np.linspace(0 + self.agent.orientation, 2* np.pi + self.agent.orientation, num_rays, endpoint=False)
        self.lidar_readings = [self.cast_ray(angle) / self.max_lidar_dist for angle in angles]
        #print(f"LiDAR readings: {self.lidar_readings}")

        state = np.array([self.relative_position[0], self.relative_position[1], 
                          relative_orientation, collision_detected]
                          + self.lidar_readings, 
                          dtype=np.float32)
        
        # Round the state to n decimal place
        state = np.round(state, 2)


        ###/// Reward calculation ///###
        max_distance_to_goal = np.linalg.norm(np.array(self.agent.initial_position) - np.array(self.goal_position))
        distance_to_goal = np.linalg.norm(np.array(self.agent.position) - np.array(self.goal_position))
        # normalized_distance_to_goal = 1 - distance_to_goal/max_distance_to_goal # value between 0-1
        old_distance_to_goal = np.linalg.norm(np.array(old_position) - np.array(self.goal_position))

        # Initialize reward variables 
        reward_goal = 0.0
        reward_collision_penalty = 0.0
        reward_visit = 0.0
        reward_forward = 0.0

        # Large reward for reaching the goal
        if distance_to_goal < 15:  # Assuming a small radius around the goal
            reward_goal = 5 * self.calc_exp_decay(relative_orientation, 1) #* max_distance_to_goal / self.screen_width
            terminated = True
            print(f"Reached the goal in {self.timesteps} timesteps!")
        else:
            terminated = False

        # Time penalty
        #reward_time_penalty = -0.1 * ((self.timesteps**0.1)-1)
        reward_time_penalty = -0.005

        # Proximity reward
        # if self.old_reward_proximity:
        #     self.old_reward_proximity = self.new_reward_proximity
        # else:
        #     self.old_reward_proximity = self.calc_gaussian(distance_to_goal, 0.0001) # initialize old_reward to be same as new_reward (returns 0 reward for first step)
        # self.new_reward_proximity = self.calc_gaussian(distance_to_goal, 0.0001)
        # reward_proximity = (self.new_reward_proximity - self.old_reward_proximity) * 300
        # if reward_proximity < 0:
        #     reward_proximity = 0.0

    
        # Orientation reward
        if distance_to_goal < self.record_distance: # only give reward if agent moves closer to the goal
            reward_orientation = self.calc_gaussian_2(relative_orientation, 2) * np.sqrt((new_position[0] - old_position[0])**2 * (new_position[1] - old_position[1])**2) * 0.005 * 0.0
            reward_proximity = (-1/self.screen_width * abs(distance_to_goal) + 1) * 0.0 #self.calc_gaussian(distance_to_goal, 0.0001 / ((self.grid_width)/10)) 
            self.record_distance = distance_to_goal
        else:
            reward_orientation = 0.0
            reward_proximity = 0.0

        # Visit count penalty
        current_cell = (int(self.agent.position[0] / self.cell_size), int(self.agent.position[1] / self.cell_size))
        if current_cell not in self.visit_count:
            self.visit_count[current_cell] = 0
            reward_visit = 0
        self.visit_count[current_cell] += 1

        # Collision penalty
        if collision_detected:
            reward_collision_penalty = -0.0

        # LiDAR based reward
        current_lidar_reward = 0.0
        lidar_threshold = 20.0  # Define a threshold for considering a distance as "too close"
        for distance in self.lidar_readings:
            if distance < lidar_threshold:
                current_lidar_reward = (lidar_threshold - (self.half_height + self.half_width) / 2 - distance) * 0.0
        reward_lidar = current_lidar_reward - self.previous_lidar_reward
        self.previous_lidar_reward = current_lidar_reward  # Update for the next step

        # Breadcrumb reward
        breadcrumb_collection_radius = 25  # Define the collection radius
        reward_breadcrumb = 0.0

        if self.astar_path:
            for path in self.astar_path:
                for i, (x, y) in enumerate(path):
                    if not self.breadcrumbs_collected[i]:
                        distance_to_breadcrumb = np.linalg.norm(np.array(self.agent.position) - np.array([x, y]))
                        if distance_to_breadcrumb < breadcrumb_collection_radius:
                            reward_breadcrumb = 0.5*(sum(self.breadcrumbs_collected))  # Define the reward for collecting a breadcrumb
                            self.breadcrumbs_collected[i] = True  # Mark as collected
                            print(f"Breadcrumb {sum(self.breadcrumbs_collected)} collected")

        self.total_reward = reward_goal + reward_time_penalty + reward_proximity + reward_orientation + reward_collision_penalty + reward_visit + reward_lidar * reward_breadcrumb

        # if out of bounds, terminate
        if (self.agent.position[0] <= 0 or self.agent.position[0] >= self.screen_width) or (self.agent.position[1] <= 0 or self.agent.position[1] >= self.screen_height):
            terminated = True  
            self.total_reward -= 10

        if self.timesteps > max_episode_steps / 2 and self.total_reward < -5:
            truncated = True  # Define your truncation condition (for max steps, etc.)
            self.truncated_count += 1
            print(f"{self.truncated_count} environments truncated due to low reward")
        else:
            truncated = False
        
        # DEBUG
        #print(state)
        #self.log_reward_components(reward_goal, reward_time_penalty, reward_proximity, reward_orientation, reward_collision_penalty, reward_visit, reward_lidar, reward_breadcrumb)

        return state, self.total_reward, terminated, truncated, {}
    

    def log_reward_components(self, reward_goal, reward_time_penalty, reward_proximity, reward_orientation, reward_collision, reward_visit, reward_lidar, reward_breadcrumb):
        print(f"Reward breakdown -> Goal: {reward_goal}, Time Penalty: {reward_time_penalty}, Proximity: {reward_proximity}, Orientation: {reward_orientation}, Collision: {reward_collision}, Visit: {reward_visit}, LiDAR: {reward_lidar}, Breadcrumb: {reward_breadcrumb}")


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.total_reward = 0.0
        self.record_distance = 999999
        self.maze_grid = generate_maze(self.grid_width, self.grid_height)

        
        # Generate a random initial position within the bounds of the screen size
        screen_width_bound = self.screen_width / self.scale
        screen_height_bound = self.screen_height / self.scale

        while True:
            random_x = random.uniform(0, screen_width_bound)
            random_y = random.uniform(0, screen_height_bound)
            if not self.is_agent_collision((random_x, random_y)) and not self.is_agent_collision((random_x + 7, random_y + 7)) and not self.is_agent_collision((random_x + 7, random_y - 7)) and not self.is_agent_collision((random_x - 7, random_y - 7)) and not self.is_agent_collision((random_x - 7, random_y + 7)):
                break

        self.agent.initial_position = (random_x, random_y)

        # Reset goal position
        self.goal_position = self.generate_goal(seed)

        relative_x = self.calc_exp_decay(self.goal_position[0] - self.agent.initial_position[0], 0.01)
        relative_y = self.calc_exp_decay(self.goal_position[1] - self.agent.initial_position[1], 0.01)
        self.relative_initial_position = (relative_x, relative_y)


        # Get path to the goal from A* algorithm
        graph = astar.create_graph_from_maze(self.maze_grid, self.grid_width * 2 + 1)
        #print("graph.edges:")
        #for node, edges in graph.edges.items():
            #print(f"{node}: {edges}")
        # Start node is the node with the closest euclidian distance to the agent on spawn
        start_node = astar.find_closest_node(self.agent.initial_position, graph)
        #print(f"self.agent.initial_position node: {self.agent.initial_position}")      
        #print(f"start node: {start_node}")
        goal_node = astar.find_closest_node(self.goal_position, graph)
        #print(f"self.goal_position: {self.goal_position}")      
        #print(f"goal node: {goal_node}")
        came_from, cost_so_far = astar.a_star_search(graph, start_node, goal_node)
        self.astar_path = astar.reconstruct_path(came_from, start_node, goal_node)
        #print(f"self.astar_path: {self.astar_path}")
        self.breadcrumbs_collected = [False] * sum(len(path) for path in self.astar_path) # Initialize all as not collected

        # Also generate random initial roation 
        self.agent.initial_orientation = random.uniform(0, 2 * np.pi)

        # Relative initial orientation 
        dx = self.goal_position[0] - self.agent.initial_position[0]
        dy = self.goal_position[1] - self.agent.initial_position[1]
        # Calculate the desired angle to face the goal
        desired_orientation = np.arctan2(dy, dx)
        # Adjust the desired orientation to have 0 facing south
        desired_orientation = (desired_orientation + 1.5 * np.pi) % (2 * np.pi)

        # Calculate the difference in orientation
        orientation_diff = self.agent.initial_orientation - desired_orientation
        # print(f"orientation diff before: {orientation_diff}")
        # Adjust orientation_diff to that it is positive when goal is to left of agent and negative when right of agent, and 0 being facing the goal and +/-pi being back to the goal
        if orientation_diff > np.pi or orientation_diff < -np.pi:
            if orientation_diff  >= 0:
                orientation_diff = -(np.pi - (orientation_diff % (np.pi)))
            elif orientation_diff < 0:
                orientation_diff = orientation_diff + 2*np.pi #(np.pi + (orientation_diff % (np.pi)))

        # print(f"orientation diff after: {orientation_diff}")
        relative_initial_orientation = orientation_diff / np.pi # normalize with a linear function so that 0 maps to 0, π maps to 1, and -π maps to -1. Hopefully this gives better orientation vision to the agent.

        # if orientation_diff < 0:
        #     goal_side = 1 # if goal is to the right of the agent
        # else:
        #     goal_side = 0 # if goal is to the left of the agent

        # print(f"relative orientation: {relative_orientation}")
        # print(f"agent orientation: {self.agent.orientation}")


        self.agent.position = self.agent.initial_position
        self.agent.orientation = self.agent.initial_orientation

        self.agent.linearVelocity = (0, 0)
        self.agent.angularVelocity = 0  # Reset angular velocity
        # print(f"Agent reset to position: {self.agent.position}")

        # Reset cumulative penalties and counters
        self.timesteps = 0.0
        self.visit_count = {}  # Reset visit count
        
        num_rays = 8
        angles = np.linspace(0, np.pi + self.agent.orientation, num_rays, endpoint=False)
        # lidar_readings = []
        # for angle in angles:
        #     distance = self.cast_ray(angle)
        #     lidar_readings.append(distance)

        self.lidar_readings = [self.cast_ray(angle) / self.max_lidar_dist for angle in angles]

        initial_state = np.array([self.relative_initial_position[0], self.relative_initial_position[1], 
                          relative_initial_orientation, 0]
                          + self.lidar_readings, 
                          dtype=np.float32)
        
        # Round the initial state to n decimal place
        initial_state = np.round(initial_state, 2)

        return initial_state, {}

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.font = pygame.font.Font(None, 24)  # Adjust the font size to your preference
            pygame.display.set_caption("Continuous Maze Environment")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

        self.screen.fill((0, 0, 0))  # Black background

        # Draw Maze using the maze_grid directly
        for row in self.maze_grid:
            for tile in row:
                x = (tile.coordinateX * 2 * self.cell_size)
                y = (tile.coordinateY * 2 * self.cell_size)
                # Draw the center tile
                pygame.draw.rect(self.screen, (255, 255, 255), (x + self.cell_size, y + self.cell_size, self.cell_size, self.cell_size))
                # Draw connections
                if "N" in tile.connectTo:
                    pygame.draw.rect(self.screen, (255, 255, 255), (x + self.cell_size, y, self.cell_size, self.cell_size))
                if "S" in tile.connectTo:
                    pygame.draw.rect(self.screen, (255, 255, 255), (x + self.cell_size, y + 2 * self.cell_size, self.cell_size, self.cell_size))
                if "W" in tile.connectTo:
                    pygame.draw.rect(self.screen, (255, 255, 255), (x, y + self.cell_size, self.cell_size, self.cell_size))
                if "E" in tile.connectTo:
                    pygame.draw.rect(self.screen, (255, 255, 255), (x + 2 * self.cell_size, y + self.cell_size, self.cell_size, self.cell_size))
        
        # Draw A* path as breadcrumbs
        if self.astar_path:
            for path in self.astar_path:
                for i, (x, y) in enumerate(path):
                    color = (0, 128, 0) if not self.breadcrumbs_collected[i] else (128, 128, 128)  # Green if not collected, grey if collected
                    pygame.draw.circle(self.screen, color, (x, y), 5)

        # Draw agent
        position = self.scale * np.array(self.agent.position)
        # print(f"Rendering agent at position: {position}")  # Debugging line

        # Rotate and translate vertices to the global frame
        global_vertices = []
        for vertex in self.agent_vertices:
            x_local, y_local = vertex
            x_global = x_local * np.cos(self.agent.orientation) - y_local * np.sin(self.agent.orientation) + position[0]
            y_global = x_local * np.sin(self.agent.orientation) + y_local * np.cos(self.agent.orientation) + position[1]
            global_vertices.append((x_global, y_global))

        # Check if the agent position is within the screen bounds
        if 0 <= position[0] <= self.screen_width and 0 <= position[1] <= self.screen_height:
            # pygame.draw.circle(self.screen, (255, 0, 0), position, int(self.agent.fixtures[0].shape.radius * self.scale))
            pygame.draw.polygon(self.screen, (255, 0, 0), global_vertices)
        else:
            print(f"Agent position {position} is out of bounds")

        # Render goal 
        pygame.draw.circle(self.screen, (0,255,0), self.goal_position, radius = 15)

        # Render LiDAR rays
        # num_rays = 9
        # angles = np.linspace(0 + self.agent.orientation, np.pi + self.agent.orientation, num_rays, endpoint=True)
        # for angle in angles:
        #     start_point = self.scale * np.array(self.agent.position)
        #     max_distance = self.cast_ray(angle)
        #     end_point = start_point + self.scale * max_distance * np.array([np.cos(angle), np.sin(angle)])
        #     pygame.draw.line(self.screen, (0, 0, 255), start_point, end_point, 1)  # Blue lines for LiDAR rays

        # Render LiDAR rays
        num_rays = 8
        angles = np.linspace(0 + self.agent.orientation, 2*np.pi + self.agent.orientation, num_rays, endpoint=False)
        for angle, distance in zip(angles, self.lidar_readings):
            start_point = self.scale * np.array(self.agent.position)
            end_point = start_point + self.scale * distance * self.max_lidar_dist * np.array([np.cos(angle), np.sin(angle)])
            pygame.draw.line(self.screen, (0, 0, 255), start_point, end_point, 1)  # Blue lines for LiDAR rays

        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

        if self.render_mode == 'rgb_array':
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
        elif self.render_mode == 'human':
            return None

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

from gymnasium.envs.registration import register

try:
    register(
        id='ContinuousMazeEnv-v1',
        entry_point='continuous_maze_env:ContinuousMazeEnv',
        max_episode_steps=max_episode_steps,
    )
    print("Environment `ContinuousMazeEnv-v1` registered successfully.")
except Exception as e:
    print(f"Failed to register environment: {e}")
