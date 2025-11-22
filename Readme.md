Autonomous Delivery Agent
The goal of the agent is to quantitatively assess the performance (path cost, execution time, nodes expanded) of BFS, UCS, and A* across various map sizes and complexities. Show how an agent can adapt and re-plan its path in real-time when faced with moving obstacles.

This project demonstrates the basics of:

Pathfinding: How robots figure out where to go.

State Management: How the robot knows if it is carrying a package or is empty.

Environment Interaction: Moving within a defined space.

🚀 Key Features
Smart Navigation: The agent calculates the shortest path from start to finish (using algorithms like BFS or A*).

Obstacle Avoidance: The agent detects walls and blocks, routing around them automatically.

Dynamic Grid: You can easily change the size of the city and the location of obstacles.

Visual Output: Watch the agent move step-by-step through the console or terminal.

Technologies used:-
1. Python 3.12
2. Google Colab
3. GitHub
4. Libraries :- 
   2.1. pandas
   2.2. matplotlib
   2.3. seaborn
   2.4. numpy
   2.5. collections
   2.6. heapq
   2.7. time
   2.8. random

Steps to run the project:- 
1. Copy the codes mentioned in the agent_pathfinding_env.py and main.py file in you VS Code or Google Colab
2. Make sure that all the required libraries are pre installed 
3. If not then install the required libraries
4. Run the main.py file

For testing the delivery agent use the following code:-

import unittest
from agent import find_path

class TestPathfinding(unittest.TestCase):
    def test_simple_path(self):
        # A simple 3x3 grid with no obstacles
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        start = (0, 0)
        end = (0, 2)
        
        # We expect the path to be 2 steps to the right
        expected_path = [(0, 0), (0, 1), (0, 2)]
        self.assertEqual(find_path(grid, start, end), expected_path)

    def test_obstacle_avoidance(self):
        # A grid with a wall in the middle
        grid = [[0, 1, 0], [0, 1, 0], [0, 0, 0]]
        start = (0, 0)
        end = (0, 2)
        
        # Path should go down and around the wall
        path = find_path(grid, start, end)
        self.assertNotIn((0, 1), path) # Ensure it didn't walk through the wall
