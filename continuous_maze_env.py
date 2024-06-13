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
    newMaze.makeMazeGrowTree(weightHigh = 99, weightLow = 97) #higher both weights are, the harder the maze is to solve
    newMaze.makeMazeBraided(-1) #removes all dead ends
    # newMaze.makeMazeBraided(7) #or introduces random loops, by taking a percentage of tiles that will have additional connections
    mazeImageBW = newMaze.makePP()
    # mazeImageBW.show() #can or can not work, see Pillow documentation. For debuging only

    return newMaze.get_grid()

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + 2 + 1 + 2 + 9,), dtype=np.float32)

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
            linearDamping=0.5, 
            angularDamping=0.5)
        # self.agent.CreateCircleFixture(radius=10, density=1.0, friction=0.3)

        # Define the vertices of a rectangle centered at the agent's position
        self.half_width = 7
        self.half_height = 10
        self.agent_vertices = [(-self.half_width, -self.half_height), (self.half_width, -self.half_height), (self.half_width, self.half_height), (-self.half_width, self.half_height)]
        self.agent.CreatePolygonFixture(vertices=self.agent_vertices, density=1.0, friction=0.3)
        # print(f"Agent created at position: {self.agent.position}")  # Debugging line

        # setting constants for angular rotation physics
        self.agent.mass = 100
        self.agent.moment_of_inertia = self.agent.mass * (self.agent.fixtures[0].shape.radius ** 2) / 2

        self.maze_grid = generate_maze(self.grid_width, self.grid_height)

        self.screen = None
        self.clock = None
        self.viewer = None
        self.font = None
        self.agent_orientation = 0.0
        self.timesteps = 0.0
        self.goal_position = self.generate_goal(self.seed)
        self.previous_lidar_reward = 0.0
        self.visit_count = {}

        self.reset()

    def generate_goal(self, seed):
        if seed is not None:
            random.seed(seed)
        while True:
            # Generate a random position within the maze boundaries
            goal_x = random.uniform(0, self.screen_width / self.scale)
            goal_y = random.uniform(0, self.screen_height / self.scale)

            # Define the upper left quadrant boundaries
            upper_left_quadrant_x = self.screen_width / self.scale / 2
            upper_left_quadrant_y = self.screen_height / self.scale / 2
            
            # Check if the goal is not in the upper left quadrant and not in a wall
            if (goal_x >= upper_left_quadrant_x or goal_y >= upper_left_quadrant_y) and not self.is_agent_collision((goal_x, goal_y)):
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
            x_global = x_local * np.cos(self.agent_orientation) - y_local * np.sin(self.agent_orientation) + position[0]
            y_global = x_local * np.sin(self.agent_orientation) + y_local * np.cos(self.agent_orientation) + position[1]
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
        left, right = 0, 100.0  # Max distance of 100 units
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

    def step(self, action):
        # print(f"Received action: {action}, shape: {np.shape(action)}")
        self.timesteps += 1
        # Store the old position in case we need to revert
        old_position = np.array(self.agent.position)

        forward_velocity = action[0] * 30
        side_velocity = action[1] * 30
        # angular_velocity = action[2] * 30
        
        # Apply the action to the agent
        self.agent.linearVelocity = ((forward_velocity * np.cos(self.agent_orientation) - side_velocity * np.sin(self.agent_orientation)), #x
                                     (forward_velocity * np.sin(self.agent_orientation) + side_velocity * np.cos(self.agent_orientation))) #y
        self.agent.angularVelocity = action[2] * 30  # Apply the rotation action     
        self.agent.torque = self.agent.angularVelocity * self.agent.moment_of_inertia
        # print(f"orientation is {self.agent_orientation}") 

        # Incremental rotation checking
        rotation_steps = 10
        incremental_rotation = self.agent.angularVelocity * (0.2 / 60) / rotation_steps
        collision_detected = False
        for _ in range(rotation_steps):
            self.agent_orientation += incremental_rotation
            if self.is_agent_collision(self.agent.position):
                self.agent_orientation -= incremental_rotation
                self.agent.angularVelocity = 0
                collision_detected=True
                break

        # Update agent orientation
        # self.agent_orientation += self.agent.angularVelocity * (1.0 / 60)

        # Step the Box2D world
        self.world.Step(10.0 / 60, 6, 2)

        # Get the new state
        new_position = np.array(self.agent.position) # make sure pos is np.array 

        # Check for collisions with walls
        if not self.is_agent_collision(new_position):
            self.agent_position = new_position
        else:
            # If collision, revert to old position
            self.agent.position = old_position
            self.agent.linearVelocity = (0, 0)
            collision_detected = True
            # print("collision detected")

        self.agent.ApplyTorque(self.agent.torque, wake=True)

        # Perform ray casting
        # start_time = time.time()
        num_rays = 9
        angles = np.linspace(0 + self.agent_orientation, np.pi + self.agent_orientation, num_rays, endpoint=True)
        # lidar_readings = []
        # for angle in angles:
        #     lidar_reading = self.cast_ray(angle)
        #     lidar_reading = round(lidar_reading,1)
        #     lidar_readings.append(lidar_reading)
        self.lidar_readings = [self.cast_ray(angle) for angle in angles]
        # end_time = time.time()
        # print(f"Optimized lidar calculation time: {end_time - start_time} seconds")
        #print(f"LiDAR readings: {self.lidar_readings}")
        '''
        # Create a text surface
        text = self.font.render(f"LiDAR readings: {lidar_readings}", True, (255, 255, 255))
        # Draw the text at the bottom of the screen
        self.screen.blit(text, (10, self.screen.get_height() - text.get_height() - 10))
        # Update the display
        pygame.display.flip()
        '''
        state = np.array([self.agent.position[0], self.agent.position[1], 
                          self.agent.linearVelocity[0], self.agent.linearVelocity[1], 
                          self.agent_orientation,
                          self.goal_position[0], self.goal_position[1]] 
                          + self.lidar_readings, 
                          dtype=np.float32)

        # Round the state to one decimal place
        state = np.round(state, 1)
      
        # Reward calculation
        max_distance_to_goal = np.linalg.norm(np.array(self.agent.initial_position) - np.array(self.goal_position))
        distance_to_goal = np.linalg.norm(np.array(self.agent.position) - np.array(self.goal_position))
        normalized_distance_to_goal = 1 - distance_to_goal/max_distance_to_goal # value between 0-1
        old_distance_to_goal = np.linalg.norm(np.array(old_position) - np.array(self.goal_position))
        # Initialize reward variables 
        reward_goal = 0.0
        reward_collision_penalty = 0.0
        reward_visit = 0.0
        reward_forward = 0.0

        # Large reward for reaching the goal
        if distance_to_goal < 10:  # Assuming a small radius around the goal
            reward_goal = 1000
            terminated = True
            print(f"Reached the goal in {self.timesteps} timesteps!")
        else:
            terminated = False

        # Time penalty
        reward_time_penalty = -0.001

        # Reward for moving closer to the goal
        reward_proximity = (old_distance_to_goal -distance_to_goal)
        #print(f"Reward Proximity: {reward_proximity}, Distance to Goal: {distance_to_goal}")

        # Penalty for collisions
        if collision_detected:
            reward_collision_penalty = -0.1
        #print(f"Collision Penalty: {reward_collision_penalty}")

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
                current_lidar_reward = (lidar_threshold - (self.half_height + self.half_width) / 2 - distance) * 0.05
        reward_lidar = current_lidar_reward - self.previous_lidar_reward
        self.previous_lidar_reward = current_lidar_reward  # Update for the next step
        #print(f"LiDAR Reward Difference: {reward_lidar}")

        # Move forward reward
        if action[1] > 0:
            reward_forward = 0.0 # Small reward to encourage the agent to move forward
        #print(f"Reward forward: {reward_forward}")

        

        total_reward = reward_goal + reward_time_penalty + reward_proximity + reward_collision_penalty + reward_visit + reward_lidar + reward_forward

        truncated = False  # Define your truncation condition (for max steps, etc.)

        # Log reward components
        #self.log_reward_components(reward_goal, reward_time_penalty, reward_proximity, reward_collision_penalty, reward_visit, reward_lidar, reward_forward)

        return state, total_reward, terminated, truncated, {}
    
    def log_reward_components(self, reward_goal, reward_time_penalty, reward_proximity, reward_collision, reward_visit, reward_lidar, reward_forward):
        print(f"Reward breakdown -> Goal: {reward_goal}, Time Penalty: {reward_time_penalty}, Proximity: {reward_proximity}, Collision: {reward_collision}, Visit: {reward_visit}, LiDAR: {reward_lidar}, Forward: {reward_forward}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # initial_position = (((self.cell_size/4)) / self.scale, ((self.cell_size/4)) / self.scale)
        self.agent.initial_position = (50,50)
        self.agent.linearVelocity = (0, 0)
        self.agent.angularVelocity = 0  # Reset angular velocity
        self.agent_orientation = 0.0  # Reset orientation
        # print(f"Agent reset to position: {self.agent.position}")
        # Reset goal position
        self.goal_position = self.generate_goal(seed)

        # Reset cumulative penalties and counters
        self.timesteps = 0.0
        self.visit_count = {}  # Reset visit count
        
        num_rays = 9
        angles = np.linspace(0, np.pi + self.agent_orientation, num_rays, endpoint=True)
        # lidar_readings = []
        # for angle in angles:
        #     distance = self.cast_ray(angle)
        #     lidar_readings.append(distance)
        self.lidar_readings = [self.cast_ray(angle) for angle in angles]


        initial_state = np.array([self.agent.initial_position[0], self.agent.initial_position[1], 
                                  self.agent.linearVelocity[0], self.agent.linearVelocity[1], 
                                  self.agent_orientation,
                                  self.goal_position[0], self.goal_position[1]] 
                                  + self.lidar_readings, 
                                  dtype=np.float32)
        
        # Round the initial state to one decimal place
        initial_state = np.round(initial_state, 1)

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
            x_global = x_local * np.cos(self.agent_orientation) - y_local * np.sin(self.agent_orientation) + position[0]
            y_global = x_local * np.sin(self.agent_orientation) + y_local * np.cos(self.agent_orientation) + position[1]
            global_vertices.append((x_global, y_global))

        # Check if the agent position is within the screen bounds
        if 0 <= position[0] <= self.screen_width and 0 <= position[1] <= self.screen_height:
            # pygame.draw.circle(self.screen, (255, 0, 0), position, int(self.agent.fixtures[0].shape.radius * self.scale))
            pygame.draw.polygon(self.screen, (255, 0, 0), global_vertices)
        else:
            print(f"Agent position {position} is out of bounds")

        # Render goal 
        pygame.draw.circle(self.screen, (0,255,0), self.goal_position, radius = 10)

        # Render LiDAR rays
        # num_rays = 9
        # angles = np.linspace(0 + self.agent_orientation, np.pi + self.agent_orientation, num_rays, endpoint=True)
        # for angle in angles:
        #     start_point = self.scale * np.array(self.agent.position)
        #     max_distance = self.cast_ray(angle)
        #     end_point = start_point + self.scale * max_distance * np.array([np.cos(angle), np.sin(angle)])
        #     pygame.draw.line(self.screen, (0, 0, 255), start_point, end_point, 1)  # Blue lines for LiDAR rays

        # Render LiDAR rays
        num_rays = 9
        angles = np.linspace(0 + self.agent_orientation, np.pi + self.agent_orientation, num_rays, endpoint=True)
        for angle, distance in zip(angles, self.lidar_readings):
            start_point = self.scale * np.array(self.agent.position)
            end_point = start_point + self.scale * distance * np.array([np.cos(angle), np.sin(angle)])
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
        max_episode_steps=512,
    )
    print("Environment `ContinuousMazeEnv-v1` registered successfully.")
except Exception as e:
    print(f"Failed to register environment: {e}")
