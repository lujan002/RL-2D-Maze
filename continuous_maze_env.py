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
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Continuous observation space: [position_x, position_y, velocity_x, velocity_y, orientation, goal_x, goal_y, lidar_array]
        # Continuous observation space: [position_relative_x, position_relative_y, orientation, collision, lidar_array]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + 1 + 1 + 9,), dtype=np.float32)

        # Pygame initialization
        self.cell_size = 40  # Each cell is _x_ pixels
        self.grid_width = 10  # _ cells wide
        self.grid_height = 10  # _ cells tall
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
        self.half_width = 7
        self.half_height = 10
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
        # Calculate the Gaussian-like relative position of the agent with respect to the goal
        return np.exp(-lambda_val * abs(x))
        
    def calculate_gaussian_orientation(self, agent_orientation, agent_position, goal_position):
        #Calculate the Gaussian-like relative orientation of the agent with respect to the goal.

        dx = goal_position[0] - agent_position[0]
        dy = goal_position[1] - agent_position[1]
        
        # Calculate the desired angle to face the goal
        desired_orientation = np.arctan2(dy, dx)
        
        # Calculate the difference in orientation
        orientation_diff = desired_orientation - agent_orientation
        
        # Normalize the orientation difference to the range [0, 2*pi)
        orientation_diff = (orientation_diff + 2 * np.pi) % (2 * np.pi) - np.pi/2
        
        # Apply the Gaussian-like function
        lambda_value = 4 * np.log(2) / np.pi**2 # Lambda for with gaussian dist (x^2)
        relative_orientation = np.exp(-lambda_value * orientation_diff**2)
        
        return relative_orientation
    
    def step(self, action):
        # Select only one action to be active
        # max_action_index = np.argmax(np.abs(action))
        # new_action = np.zeros_like(action)
        # new_action[max_action_index] = action[max_action_index]
        # action = new_action

        # print(f"Received action: {action}")
        self.timesteps += 1
        # print(f"Timestep: {self.timesteps}")

        '''
        # # Store the old position in case we need to revert
        # old_position = np.array(self.agent.position)

        # side_velocity = action[0] * 100
        # forward_velocity = action[1] * 100
        
        # # Apply the action to the agent
        # self.agent.linearVelocity = ((side_velocity * np.cos(self.agent.orientation) - forward_velocity * np.sin(self.agent.orientation)), #x
        #                              (side_velocity * np.sin(self.agent.orientation) + forward_velocity * np.cos(self.agent.orientation))) #y
        # self.agent.linearVelocity_np = np.array([self.agent.linearVelocity.x, self.agent.linearVelocity.y])
        # self.agent.angularVelocity = action[2] * 100  # Apply the rotation action     s
        # self.agent.torque = self.agent.angularVelocity * self.agent.moment_of_inertia
        # self.agent.force = self.agent.linearVelocity * self.agent.moment_of_inertia
        # # print(f"orientation is {self.agent.orientation}") 
        # print(f"Linear Velocity: {self.agent.linearVelocity_np}")
        # print(f"Angular Velocity: {self.agent.angularVelocity}")
        # print(f"Force: {self.agent.force}")
        # print(f"Torque: {self.agent.torque}")
        # # Incremental rotation checking
        # rotation_steps = 10
        # incremental_rotation = self.agent.angularVelocity * (0.2 / 60) / rotation_steps
        # collision_detected = False
        # for _ in range(rotation_steps):
        #     self.agent.orientation += incremental_rotation
        #     if self.is_agent_collision(self.agent.position):
        #         self.agent.orientation -= incremental_rotation
        #         self.agent.angularVelocity = 0
        #         collision_detected=True
        #         break

        ##Step the Box2D world
        # self.world.Step(1.0 / 60, 6, 2)

        # # Get the new state
        # new_position = np.array(self.agent.position) # make sure pos is np.array 

        # self.agent.ApplyTorque(self.agent.torque, wake=True)
        # self.agent.ApplyForceToCenter(self.agent.force, wake=True)
'''
        # Store the old position for potential collision checks
        old_position = np.array(self.agent.position)
        old_orientation = self.agent.orientation
        
        # Directly update the position based on the action
        side_velocity = action[0] * 10  # Adjust scaling factor as needed
        forward_velocity = action[1] * 10  # Adjust scaling factor as needed
        
        # Calculate the new position
        dx = side_velocity * np.cos(self.agent.orientation) - forward_velocity * np.sin(self.agent.orientation)
        dy = side_velocity * np.sin(self.agent.orientation) + forward_velocity * np.cos(self.agent.orientation)
        
        new_position = old_position + np.array([dx, dy])
        
        # Update the orientation
        dtheta = action[2] * 0.3  # Adjust scaling factor as needed

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
        
        old_relative_position = self.relative_position
        # # Calculate the relative position to the goal using Gaussian function
        relative_x = self.calc_gaussian(self.goal_position[0] - self.agent.position[0], 0.01)
        relative_y = self.calc_gaussian(self.goal_position[1] - self.agent.position[1], 0.01)
        self.relative_position = (relative_x, relative_y)

        # # Calculate the relative orientation to the goal using Gaussian function
        # # Normalize the orientation to be within the range [0, 2*pi)
        # self.agent.orientation = self.agent.orientation % (2 * np.pi)
        # if self.agent.orientation < 0:
        #     self.agent.orientation += 2 * np.pi

        # # Calculate the angle to the goal
        # goal_direction = np.arctan2(self.goal_position[1] - self.agent.position[1], self.goal_position[0] - self.agent.position[0])
        # relative_orientation = (self.agent.orientation - goal_direction) % (2 * np.pi)
		# # Normalize orientation to [0, 1]
        # relative_orientation = relative_orientation / (2 * np.pi)

        relative_orientation = self.calculate_gaussian_orientation(self.agent.orientation, self.agent.position, self.goal_position)

        # Perform ray casting
        num_rays = 9
        angles = np.linspace(0 + self.agent.orientation, np.pi + self.agent.orientation, num_rays, endpoint=True)

        self.lidar_readings = [self.cast_ray(angle) / self.max_lidar_dist for angle in angles]
        #print(f"LiDAR readings: {self.lidar_readings}")
 
        state = np.array([self.relative_position[0], self.relative_position[1], 
                          relative_orientation, collision_detected]
                          + self.lidar_readings, 
                          dtype=np.float32)
        
        # Round the state to n decimal place
        state = np.round(state, 3)

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
            reward_goal = 3
            terminated = True
            print(f"Reached the goal in {self.timesteps} timesteps!")
        else:
            terminated = False

        # Time penalty
        reward_time_penalty = -0.1 * ((self.timesteps**0.1)-1)

        # Proximity reward
        if self.old_reward_proximity:
            self.old_reward_proximity = self.new_reward_proximity
        else:
            self.old_reward_proximity = self.calc_gaussian(distance_to_goal, 0.0001) # initialize old_reward to be same as new_reward (returns 0 reward for first step)
        self.new_reward_proximity = self.calc_gaussian(distance_to_goal, 0.0001)
        reward_proximity = (self.new_reward_proximity - self.old_reward_proximity) * 300
        if reward_proximity < 0:
            reward_proximity = 0.0

        # Orientation reward
        reward_orientation = self.agent.orientation 

        # Collision penalty
        if collision_detected:
            reward_collision_penalty = -0.0

        # Visit count penalty
        current_cell = (int(self.agent.position[0] / self.cell_size), int(self.agent.position[1] / self.cell_size))
        if current_cell not in self.visit_count:
            self.visit_count[current_cell] = 0
            reward_visit = 0
        self.visit_count[current_cell] += 1

        # LiDAR based reward
        current_lidar_reward = 0.0
        lidar_threshold = 20.0  # Define a threshold for considering a distance as "too close"
        for distance in self.lidar_readings:
            if distance < lidar_threshold:
                current_lidar_reward = (lidar_threshold - (self.half_height + self.half_width) / 2 - distance) * 0.0
        reward_lidar = current_lidar_reward - self.previous_lidar_reward
        self.previous_lidar_reward = current_lidar_reward  # Update for the next step

        self.total_reward = reward_goal + reward_time_penalty + reward_proximity + reward_orientation + reward_collision_penalty + reward_visit + reward_lidar

        truncated = False  # Define your truncation condition (for max steps, etc.)

        # DEBUG
        print(state)
        #self.log_reward_components(reward_goal, reward_time_penalty, reward_proximity, reward_orientation, reward_collision_penalty, reward_visit, reward_lidar)

        return state, self.total_reward, terminated, truncated, {}
    
    def log_reward_components(self, reward_goal, reward_time_penalty, reward_proximity, reward_orientation, reward_collision, reward_visit, reward_lidar):
        print(f"Reward breakdown -> Goal: {reward_goal}, Time Penalty: {reward_time_penalty}, Proximity: {reward_proximity}, Orientation: {reward_orientation}, Collision: {reward_collision}, Visit: {reward_visit}, LiDAR: {reward_lidar}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.total_reward = 0.0
        self.maze_grid = generate_empty_maze(self.grid_width, self.grid_height)

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

        relative_x = self.calc_gaussian(self.goal_position[0] - self.agent.initial_position[0], 0.01)
        relative_y = self.calc_gaussian(self.goal_position[1] - self.agent.initial_position[1], 0.01)
        self.relative_initial_position = (relative_x, relative_y)

        # Also generate random initial roation 
        self.agent.initial_orientation = random.uniform(0, 2 * np.pi)
        relative_initial_orientation = self.calculate_gaussian_orientation(self.agent.initial_orientation, self.agent.initial_position, self.goal_position)

        self.agent.position = self.agent.initial_position
        self.agent.orientation = self.agent.initial_orientation

        self.agent.linearVelocity = (0, 0)
        self.agent.angularVelocity = 0  # Reset angular velocity
        # print(f"Agent reset to position: {self.agent.position}")

        # Reset cumulative penalties and counters
        self.timesteps = 0.0
        self.visit_count = {}  # Reset visit count
        
        num_rays = 9
        angles = np.linspace(0, np.pi + self.agent.orientation, num_rays, endpoint=True)
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
        initial_state = np.round(initial_state, 3)

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
        num_rays = 9
        angles = np.linspace(0 + self.agent.orientation, np.pi + self.agent.orientation, num_rays, endpoint=True)
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
        max_episode_steps=256,
    )
    print("Environment `ContinuousMazeEnv-v1` registered successfully.")
except Exception as e:
    print(f"Failed to register environment: {e}")
