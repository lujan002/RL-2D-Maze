import heapq
from Maze import Maze
import numpy as np
from collections import deque

class Graph:
    def __init__(self):
        self.edges = {}
        self.weights = {}

    def neighbors(self, node):
        return self.edges.get(node, [])

    def cost(self, from_node, to_node):
        return self.weights[(from_node, to_node)]

def flood_fill(graph, goal):
    distances = {node: float('inf') for node in graph.edges}
    queue = deque([goal])
    distances[goal] = 0

    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        for neighbor in graph.neighbors(current):
            if current_distance + graph.cost(current, neighbor) < distances[neighbor]:
                distances[neighbor] = current_distance + graph.cost(current, neighbor)
                queue.append(neighbor)

    return distances

def heuristic(neighbor, distances):
	print("distances:")
	for node, distances in distances.items():
		print(f"{node}: {distances}")

	print(f"neightbor: {neighbor}")
	return distances[neighbor[1], neighbor[0]]

def a_star_search(graph, start, goal):
	queue = []
	heapq.heappush(queue, (0, start))
	came_from = {}
	cost_so_far = {}
	came_from[start] = None
	cost_so_far[start] = 0
	distances = flood_fill(graph, goal)
	while queue:
		current = heapq.heappop(queue)[1]

		if current == goal:
			break

		for neighbor in graph.neighbors(current):
			new_cost = cost_so_far[current] + graph.cost(current, neighbor)
			if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
				cost_so_far[neighbor] = new_cost
				priority = new_cost + distances[neighbor]
				heapq.heappush(queue, (priority, neighbor))
				came_from[neighbor] = current

		print(f"Current: {current}, Cost so far: {cost_so_far}")

	return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()

	# Remove the first and last nodes
    if len(path) > 1:
        path = path[1:-1]

    print(f"path: {path}")
    return path

# def create_graph_from_maze(maze_grid):
#     graph = Graph()
#     print(f"maze_grid: {maze_grid}")
#     for row in maze_grid:
#         for tile in row:
#             node = (tile.coordinateX, tile.coordinateY) 
#             graph.edges[node] = []
#             for direction in tile.connectTo:
#                 if direction == "N":
#                     neighbor = (tile.coordinateX, tile.coordinateY - 1)
#                 elif direction == "S":
#                     neighbor = (tile.coordinateX, tile.coordinateY + 1)
#                 elif direction == "W":
#                     neighbor = (tile.coordinateX - 1, tile.coordinateY)
#                 elif direction == "E":
#                     neighbor = (tile.coordinateX + 1, tile.coordinateY)
#                 if 0 <= neighbor[0] < len(maze_grid[0]) and 0 <= neighbor[1] < len(maze_grid):
#                     graph.edges[node].append(neighbor)
#                     graph.weights[(node, neighbor)] = 1  # Assuming uniform cost

	# # Apply the transformation to all edges and weights
    # transformed_edges = {}
    # transformed_weights = {}
    # for node in graph.edges:
    #     transformed_node = (node[0] * 40 + 20, node[1] * 40 + 20)
    #     transformed_edges[transformed_node] = [
    #         (neighbor[0] * 40 + 20, neighbor[1] * 40 + 20) for neighbor in graph.edges[node]
    #     ]
    #     for neighbor in graph.edges[node]:
    #         transformed_neighbor = (neighbor[0] * 40 + 20, neighbor[1] * 40 + 20)
    #         transformed_weights[(transformed_node, transformed_neighbor)] = graph.weights[(node, neighbor)]
    # graph.edges = transformed_edges
    # graph.weights = transformed_weights

#     return graph

def create_graph_from_maze(maze_grid, grid_size):
	graph = Graph()
	print(f"maze grid {maze_grid}")
	
	# Initialize the (2*grid_size+1) x grid_size grid with zeros
	nodes = np.zeros((grid_size, grid_size))

	for row in maze_grid:
		for tile in row:
			print(f"tile connect to: {tile.connectTo}")
			
			# Populate the center nodes
			node_x = tile.coordinateX * 2 + 1
			node_y = tile.coordinateY * 2 + 1
			nodes[node_y, node_x] = 1
					
			# Populate the nodes grid based on the tile connections
			for direction in tile.connectTo:
				if direction == "N":
					nodes[node_y - 1, node_x] = 1
				elif direction == "S":
					nodes[node_y + 1, node_x] = 1
				elif direction == "W":
					nodes[node_y, node_x - 1] = 1
				elif direction == "E":
					nodes[node_y, node_x + 1] = 1

			# Print the resulting grid for debugging
			for row in nodes:
				print(row)

	# Add nodes and edges to the graph
	for i in range(grid_size):
		for j in range(grid_size):
			if nodes[i][j] == 1:
				node = (j, i)
				graph.edges[node] = []
				if i > 0 and nodes[i - 1][j] == 1:
					neighbor = (j, i-1)
					graph.edges[node].append(neighbor)
					graph.weights[(node, neighbor)] = 1
				if i < grid_size - 1 and nodes[i + 1][j] == 1:
					neighbor = (j, i+1)
					graph.edges[node].append(neighbor)
					graph.weights[(node, neighbor)] = 1
				if j > 0 and nodes[i][j - 1] == 1:
					neighbor = (j-1, i)
					graph.edges[node].append(neighbor)
					graph.weights[(node, neighbor)] = 1
				if j < grid_size - 1 and nodes[i][j + 1] == 1:
					neighbor = (j+1, i)
					graph.edges[node].append(neighbor)
					graph.weights[(node, neighbor)] = 1

	print("graph.edges:")
	for node, edges in graph.edges.items():
		print(f"{node}: {edges}")


	# Apply the transformation to all edges and weights
	transformed_edges = {}
	transformed_weights = {}
	for node in graph.edges:
		transformed_node = (node[0] * 40 + 20, node[1] * 40 + 20)
		transformed_edges[transformed_node] = [
			(neighbor[0] * 40 + 20, neighbor[1] * 40 + 20) for neighbor in graph.edges[node]
		]
		for neighbor in graph.edges[node]:
			transformed_neighbor = (neighbor[0] * 40 + 20, neighbor[1] * 40 + 20)
			if (node, neighbor) in graph.weights:
				transformed_weights[(transformed_node, transformed_neighbor)] = graph.weights[(node, neighbor)]
			else:
				print(f"Warning: missing weight for edge ({node}, {neighbor})")
	graph.edges = transformed_edges
	graph.weights = transformed_weights

	return graph
	


def find_closest_node(position, graph):
	closest_node = None
	min_distance = float('inf')
	for node in graph.edges:
		distance = np.sqrt((node[0] - position[0]) ** 2 + (node[1] - position[1]) ** 2)
		if distance < min_distance:
			closest_node = node
			min_distance = distance
	return closest_node

# def find_closest_node(position, goal, graph):
# 	closest_agent_node = None
# 	closest_goal_node = None
# 	min_agent_distance = float('inf')
# 	min_goal_distance = float('inf')
# 	agent_to_goal_distance = np.sqrt((position[0] - goal[0]) ** 2 + (position[1] - goal[1]) ** 2)
	
# 	for node in graph.edges:
# 		node_to_goal_distance = np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)
# 		node_to_agent_distance = np.sqrt((node[0] - position[0]) ** 2 + (node[1] - position[1]) ** 2)
		
# 		if node_to_goal_distance < agent_to_goal_distance and node_to_agent_distance < min_agent_distance:
# 			closest_agent_node = node
# 			min_agent_distance = node_to_agent_distance

# 		if node_to_agent_distance < agent_to_goal_distance and node_to_goal_distance < min_goal_distance:
# 			closest_goal_node = node
# 			min_goal_distance = node_to_goal_distance
		
# 	return closest_agent_node, closest_goal_node


# # Maze generation functions
# def generate_maze(width, height):
#     newMaze = Maze(width,height)
#     newMaze.makeMazeGrowTree(weightHigh = 10, weightLow = 0) #higher both weights are, the harder the maze is to solve
#     newMaze.makeMazeBraided(-1) #removes all dead ends
#     # newMaze.makeMazeBraided(7) #or introduces random loops, by taking a percentage of tiles that will have additional connections
#     mazeImageBW = newMaze.makePP()
#     # mazeImageBW.show() #can or can not work, see Pillow documentation. For debuging only

#     return newMaze.get_grid()

# # Test the graph creation
# maze_grid = generate_maze(3, 3)  # Replace with the actual maze generation function
# graph = create_graph_from_maze(maze_grid)
# print("Graph edges:", graph.edges)

# start_node = (20, 20)  # Replace with the actual start node
# goal_node = (100, 20)  # Replace with the actual goal node
# came_from, cost_so_far = a_star_search(graph, start_node, goal_node)
# path = reconstruct_path(came_from, start_node, goal_node)
# print("A* path:", path)
