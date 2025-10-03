# -*- coding: utf-8 -*-
"""

    This  is

    Search and Heuristic Optimization for Path Planning
"""

import matplotlib.pyplot as plt
import numpy as np

# 10x10 grid (0 = free, 1 = obstacle)
grid = [[0]*10 for _ in range(10)]
obstacles = [(1,2), (2,2), (3,4), (5,5), (7,3), (8,7)]
for (x,y) in obstacles:
    grid[x][y] = 1

# helper functions
def is_valid_cell(x, y, grid):
    if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
        return grid[x][y] == 0  # 0 = free, 1 = obstacle
    return False

def get_neighbors(x, y, grid):
    neighbors = []
    moves = [(1,0), (-1,0), (0,1), (0,-1)]  # 4-connectivity
    for dx, dy in moves:
        if is_valid_cell(x+dx, y+dy, grid):
            neighbors.append((x+dx, y+dy))
    return neighbors

def calculate_distance(cell1, cell2):
    x1, y1 = cell1
    x2, y2 = cell2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

# visualization
def draw_grid(grid, path=None):
    grid = np.array(grid)
    plt.imshow(grid.T, cmap="gray_r", origin="lower")  # origin="lower" makes (0,0) bottom-left

    # mark start & goal
    plt.scatter(0, 0, c="green", s=100, label="Start")
    plt.scatter(9, 9, c="red", s=100, label="Goal")

    # draw path if provided
    if path:
        xs, ys = zip(*path)
        plt.plot(xs, ys, c="blue", linewidth=2, label="Path")

    plt.legend()
    plt.grid(True)
    plt.show()

# just test visualization without path
draw_grid(grid)

import heapq, time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

# --------- Helper Functions ---------
def is_valid_cell(x, y, grid):
    if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
        return grid[x][y] == 0
    return False

def get_neighbors(x, y, grid):
    neighbors = []
    moves = [(1,0), (-1,0), (0,1), (0,-1)]  # 4-connectivity
    for dx, dy in moves:
        if is_valid_cell(x+dx, y+dy, grid):
            neighbors.append((x+dx, y+dy))
    return neighbors

def calculate_distance(cell1, cell2):
    x1, y1 = cell1
    x2, y2 = cell2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def draw_grid(grid, path=None, title="Grid"):
    grid = np.array(grid)
    plt.imshow(grid.T, cmap="gray_r", origin="lower")

    plt.scatter(0, 0, c="green", s=100, label="Start")
    plt.scatter(9, 9, c="red", s=100, label="Goal")

    if path:
        xs, ys = zip(*path)
        plt.plot(xs, ys, c="blue", linewidth=2, label="Path")

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# --------- A* Algorithm ---------
def a_star(start, goal, grid):
    open_list = []
    heapq.heappush(open_list, (0, start))
    g_scores = {start: 0}
    parents = {start: None}

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parents[current]
            return path[::-1]

        for neighbor in get_neighbors(*current, grid):
            tentative_g = g_scores[current] + 1
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                f_score = tentative_g + calculate_distance(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))
                parents[neighbor] = current
    return None

# --------- BFS Algorithm ---------
def bfs(start, goal, grid):
    queue = deque([start])
    parents = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parents[current]
            return path[::-1]

        for neighbor in get_neighbors(*current, grid):
            if neighbor not in parents:
                parents[neighbor] = current
                queue.append(neighbor)
    return None

# --------- Run Experiments ---------
def run_experiment(obstacles, title="Obstacle Set"):
    # build grid
    grid = [[0]*10 for _ in range(10)]
    for (x,y) in obstacles:
        grid[x][y] = 1

    start, goal = (0,0), (9,9)

    print(f"\n===== {title} =====")
    draw_grid(grid, title=title + " (Obstacles Only)")

    # A*
    t1 = time.time()
    path_a = a_star(start, goal, grid)
    t2 = time.time()
    time_a = t2 - t1
    if path_a:
        print(f"A*: Path length = {len(path_a)-1}, Time = {time_a:.6f} sec")
        draw_grid(grid, path_a, title="A* Path")
    else:
        print("A*: No path found")

    # BFS
    t1 = time.time()
    path_b = bfs(start, goal, grid)
    t2 = time.time()
    time_b = t2 - t1
    if path_b:
        print(f"BFS: Path length = {len(path_b)-1}, Time = {time_b:.6f} sec")
        draw_grid(grid, path_b, title="BFS Path")
    else:
        print("BFS: No path found")

# --------- Test on 3 Obstacle Sets ---------
obstacle_sets = [
    [(1,2),(2,2),(3,4),(5,5),(7,3),(8,7)],   # Set 1
    [(2,1),(2,2),(2,3),(4,4),(6,6),(7,7)],   # Set 2
    [(1,1),(1,2),(2,2),(3,3),(4,3),(5,4)]    # Set 3
]

for i, obs in enumerate(obstacle_sets, start=1):
    run_experiment(obs, f"Obstacle Set {i}")

import math, time, heapq
import matplotlib.pyplot as plt
import numpy as np

# -------- Grid Setup --------
grid_size = 10
def make_grid(obstacles):
    grid = [[0]*grid_size for _ in range(grid_size)]
    for (x,y) in obstacles:
        grid[x][y] = 1
    return grid

start, goal = (0,0), (9,9)

# -------- Heuristic Functions --------
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def chebyshev(a, b):
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

# -------- Neighbors (with diagonals) --------
def get_neighbors(x, y, grid, allow_diagonal=True):
    moves = [(1,0), (-1,0), (0,1), (0,-1)]
    if allow_diagonal:
        moves += [(1,1),(1,-1),(-1,1),(-1,-1)]
    neighbors = []
    for dx, dy in moves:
        nx, ny = x+dx, y+dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny]==0:
            cost = math.sqrt(2) if dx!=0 and dy!=0 else 1
            neighbors.append(((nx,ny), cost))
    return neighbors

# -------- A* with customizable heuristic --------
def a_star(start, goal, grid, heuristic=euclidean, allow_diagonal=True):
    open_list = []
    heapq.heappush(open_list, (0, start))
    g_scores = {start: 0}
    parents = {start: None}

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parents[current]
            return path[::-1]

        for (neighbor, step_cost) in get_neighbors(*current, grid, allow_diagonal):
            tentative_g = g_scores[current] + step_cost
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))
                parents[neighbor] = current
    return None

# -------- Visualization --------
def draw_grid(grid, path=None, title="Grid"):
    grid = np.array(grid)
    plt.imshow(grid.T, cmap="gray_r", origin="lower")
    plt.scatter(start[0], start[1], c="green", s=100, label="Start")
    plt.scatter(goal[0], goal[1], c="red", s=100, label="Goal")
    if path:
        xs, ys = zip(*path)
        plt.plot(xs, ys, c="blue", linewidth=2, label="Path")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# -------- Test Heuristics --------
obstacles = [(1,2),(2,2),(3,4),(5,5),(7,3),(8,7)]
grid = make_grid(obstacles)

for heuristic in [manhattan, euclidean, chebyshev]:
    path = a_star(start, goal, grid, heuristic=heuristic, allow_diagonal=True)
    print(f"Heuristic: {heuristic.__name__}, Steps = {len(path)-1 if path else 'No path'}")
    draw_grid(grid, path, title=f"A* with {heuristic.__name__}")

# -------- Moving Obstacle Example --------
print("\n=== Moving Obstacle Demo ===")
moving_obstacle_path = [(4,4), (4,5), (4,6)]
for step, obs in enumerate(moving_obstacle_path, start=1):
    obstacles = [(1,2),(2,2),(3,4),(7,3),(8,7), obs]  # include moving obstacle
    grid = make_grid(obstacles)
    path = a_star(start, goal, grid, heuristic=euclidean, allow_diagonal=True)
    print(f"Step {step}: Moving obstacle at {obs}, Path steps = {len(path)-1 if path else 'No path'}")
    draw_grid(grid, path, title=f"Path with Moving Obstacle at {obs}")

