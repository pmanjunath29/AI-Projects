# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
import os
import pickle
import math


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        node = heapq.heappop(self.queue)
        nodeTuple = ()
        for i in range(len(node) - 2):
            nodeTuple = nodeTuple + (node[i],)
        nodeTuple = nodeTuple + (node[len(node) - 1],)
        return node
        #raise NotImplementedError

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """

        self.queue.remove(node)
        #raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        if type(node) == tuple:
            index = len(self.queue) + 1
            nodeTuple = ()
            for i in range(len(node) - 1):
                nodeTuple = nodeTuple + (node[i],)
            nodeTuple = nodeTuple + (index,)
            nodeTuple = nodeTuple + (node[len(node) - 1],)
            heapq.heappush(self.queue, nodeTuple)             
        else:
            heapq.heappush(self.queue, node)
        return self.queue
        #raise NotImplementedError
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start == goal:
        return []
    else:
        frontier = PriorityQueue()
        visited = {}
        parent_child_dict = {}
        frontier.append((0, start))
        visited[start] = True
        depth = 1
        while frontier.size() > 0:
            node = frontier.pop()
            node = node[len(node) - 1]
            if node == goal:
                break
            else:
                neighbors = graph.neighbors(node)
                neighbors = sorted(neighbors)
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor not in frontier:
                        if node not in parent_child_dict:
                            parent_child_dict[node] = []
                        parent_child_dict[node].append(neighbor)
                        if neighbor == goal:
                            return bfs_helper(graph, start, goal, parent_child_dict)
                        else:
                            frontier.append((depth, neighbor))
                            visited[neighbor] = True
                    depth = depth + 1
        return bfs_helper(graph, start, goal, parent_child_dict)
    raise NotImplementedError

def bfs_helper(graph, start, goal, parent_child_dict):
    path = []
    path.append(goal)
    cur = goal
    while cur != start:
        for key in parent_child_dict:
            if cur in parent_child_dict[key]:
                path.append(key)
                cur = key
    return path[::-1]
    
def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    if start == goal:
        return []
    else:
        frontier = PriorityQueue()
        visited = {}
        parent_child_dict = {}
        frontier.append((0, start))
        while frontier.size() > 0:
            node = frontier.pop()
            node = (node[0], node[2])
            if node[1] == goal:
                return ucs_helper(graph, start, goal, parent_child_dict)
            else:
                visited[node[1]] = True
                neighbors = graph.neighbors(node[1])
                depth = 1
                for neighbor in neighbors:
                    cost = node[0] + graph.get_edge_weight(node[1], neighbor)
                    if neighbor not in visited and neighbor not in frontier:
                        frontier.append((cost, neighbor))
                        parent_child_dict[(node[-1][-1], depth)] = (neighbor, cost)
                        depth = depth + 1
                    else:
                        for next in frontier:
                            if next[2] == neighbor and cost < next[0]:
                                frontier.remove(next)
                                frontier.append((cost, neighbor))
                                parent_child_dict[(node[1], depth)] = (neighbor, cost)
                                depth = depth + 1
        return ucs_helper(graph, start, goal, parent_child_dict)
        
def ucs_helper(graph, start, goal, parent_child_dict):
    path = []
    path.append(goal)
    cur = goal
    cost = float("+inf")
    for key in parent_child_dict:
        if goal == parent_child_dict[key][0] and parent_child_dict[key][1] < cost:
            cost = parent_child_dict[key][1]
    while cur != start:
        for key in parent_child_dict:
            if cur == parent_child_dict[key][0] and cost == parent_child_dict[key][1]:
                edge_cost = graph.get_edge_weight(cur, key[0])
                cost = cost - edge_cost
                path.append(key[0])
                cur = key[0]
    return path[::-1]

def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    return ((graph.nodes[v]['pos'][0] - graph.nodes[goal]['pos'][0]) ** 2 + (graph.nodes[v]['pos'][1] - graph.nodes[goal]['pos'][1]) ** 2) ** (1 / 2)

def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.
    
    See README.md for exercise description.
    
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
            
    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    # Ensure start and goal state are not the same, else return empty list
    if start == goal:
        return []
    else:
        frontier = PriorityQueue()
        visited = {}
        frontier.append((heuristic(graph, start, goal), 0, start))
        parent_child_dict = {}
        while frontier.size() > 0:
            node = frontier.pop()
            if node[3] == goal:
                return a_star_helper(graph, start, goal, parent_child_dict, heuristic)
            visited[node[3]] = True
            neighbors = graph.neighbors(node[3])
            depth = 1
            for neighbor in neighbors:
                edge_weight = graph.get_edge_weight(node[3], neighbor) + node[1]
                total_dist = edge_weight + heuristic(graph, neighbor, goal)
                if neighbor not in visited and neighbor not in frontier:
                    frontier.append((total_dist, edge_weight, neighbor)) 
                    parent_child_dict[(node[3], depth)] = (total_dist, edge_weight, neighbor)
                    depth = depth + 1
                else:
                    for next in frontier:
                        if next[3] == neighbor and next[0] > total_dist:
                            frontier.remove(next)
                            frontier.append((total_dist, edge_weight, neighbor))
                            visited[neighbor] = True
                            parent_child_dict[(node[3], depth)] = (total_dist, edge_weight, neighbor)
                            depth = depth + 1
        return a_star_helper(graph, start, goal, parent_child_dict, heuristic)

def a_star_helper(graph, start, goal, parent_child, heuristic):
    path = []
    path.append(goal)
    cur = goal
    cur_dist = float("+inf")
    for key in parent_child:
        if parent_child[key][2] == goal and parent_child[key][0] < cur_dist:
            cur_dist = parent_child[key][0]
    while cur != start:
        for key in parent_child:
            if cur == parent_child[key][2] and abs(cur_dist - parent_child[key][0]) < 0.1:
                edge_weight = graph.get_edge_weight(key[0], parent_child[key][2])
                cur_straight_line_dist = heuristic(graph, key[0], goal)
                next_straight_line_dist = heuristic(graph, parent_child[key][2], goal)
                cur_dist = cur_dist - edge_weight - next_straight_line_dist + cur_straight_line_dist
                path.append(key[0])
                cur = key[0]
    return path[::-1]

def bidirectional_ucs(graph, start, goal):
    if start == goal:
        return []
    start_pq = PriorityQueue()
    start_pq.append((0, start))
    goal_pq = PriorityQueue()
    goal_pq.append((0, goal))
    start_prev_dict = {}
    goal_prev_dict = {}
    start_prev_dict[start] = (0, None)
    goal_prev_dict[goal] = (0, None)
    solution_node = None
    optimacy_threshold = float("+inf")
    start_pq_top_cost = 0
    goal_pq_top_cost = 0
    while start_pq_top_cost + goal_pq_top_cost < optimacy_threshold and (start_pq.size() > 0 or goal_pq.size() > 0):
        if start_pq.size() > 0:
            tuple = bidirectional_ucs_helper(graph, start_pq, start_prev_dict, goal_prev_dict, solution_node, optimacy_threshold)
            solution_node = tuple[0]
            optimacy_threshold = tuple[1]
        if goal_pq.size() > 0:
            tuple = bidirectional_ucs_helper(graph, goal_pq, goal_prev_dict, start_prev_dict, solution_node, optimacy_threshold)
            solution_node = tuple[0]
            optimacy_threshold = tuple[1]
        start_pq_top_cost = start_pq.top()[0]
        goal_pq_top_cost = goal_pq.top()[0]
    if solution_node == None:
        return []
    return bidirectional_ucs_helper2(graph, solution_node, start, goal, start_prev_dict, goal_prev_dict)

def bidirectional_ucs_helper(graph, pq, prev_dict1, prev_dict2, solution_node, optimacy_threshold):
    node = pq.pop()
    node = (node[0], node[2])
    neighbors = [n for n in graph.neighbors(node[1])]
    frontier_nodes = []
    for n in neighbors:
        frontier_nodes.append(node[1])
    union = list(set().union(frontier_nodes, prev_dict1.keys()))
    for n in neighbors:
        if n not in union or n not in prev_dict1 or node[0] + graph.get_edge_weight(node[1], n) < prev_dict1[n][0]:
            cost = node[0] + graph.get_edge_weight(node[1], n)
            prev_dict1[n] = (cost, node[1])
            pq.append((cost, n))
            if n in prev_dict2 and prev_dict1[n][0] + prev_dict2[n][0] < optimacy_threshold:
                solution_node = n
                optimacy_threshold = prev_dict1[n][0] + prev_dict2[n][0]
    return (solution_node, optimacy_threshold)

def bidirectional_ucs_helper2(graph, node, start, goal, prev_dict1, prev_dict2):
    path = []
    cur = node
    if node != start:
        while cur != start:
            next = prev_dict1[cur][1]
            path.append(next)
            cur = next
    path = path[::-1]
    path.append(node)
    if node == goal:
        return path
    cur = node
    while cur != goal:
        next = prev_dict2[cur][1]
        path.append(next)
        cur = next
    return path

def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError

def bidirectional_a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    if start == goal:
        return []
    start_pq = PriorityQueue()
    start_pq.append((0, start))
    goal_pq = PriorityQueue()
    goal_pq.append((0, goal))
    start_prev_dict = {}
    goal_prev_dict = {}
    start_prev_dict[start] = (0, None)
    goal_prev_dict[goal] = (0, None)
    solution_node = None
    optimacy_threshold = float("+inf")
    start_pq_top_cost = 0
    goal_pq_top_cost = 0
    while (start_pq_top_cost < optimacy_threshold and goal_pq_top_cost < optimacy_threshold) and (start_pq.size() > 0 or goal_pq.size() > 0):
        if start_pq.size() > 0:
            tuple = bidirectional_a_star_helper(graph, start, goal, start_pq, start_prev_dict, goal_prev_dict, solution_node, optimacy_threshold, heuristic, True)
            #print("We in If 1")
            solution_node = tuple[0]
            #print(solution_node)
            optimacy_threshold = tuple[1]
        if goal_pq.size() > 0:
            tuple = bidirectional_a_star_helper(graph, start, goal, goal_pq, goal_prev_dict, start_prev_dict, solution_node, optimacy_threshold, heuristic, False)
            #print("We in If 2")
            solution_node = tuple[0]
            #print(solution_node)
            optimacy_threshold = tuple[1]
        start_pq_top_cost = start_pq.top()[0]
        goal_pq_top_cost = goal_pq.top()[0]
    if solution_node == None:
        return []
    return bidirectional_a_star_helper2(graph, solution_node, start, goal, start_prev_dict, goal_prev_dict)

def bidirectional_a_star_helper(graph, start, goal, pq, prev_dict1, prev_dict2, solution_node, optimacy_threshold, heuristic, check):
    node = pq.pop()
    node = (node[0], node[2])
    #print("Node popped")
    #print(node)
    neighbors = [n for n in graph.neighbors(node[1])]
    frontier_nodes = []
    for n in neighbors:
        frontier_nodes.append(node[1])
    union = list(set().union(frontier_nodes, prev_dict1.keys()))
    for n in neighbors:
        #print("Inside the for loop")
        if n not in union or n not in prev_dict1 or node[0] + graph.get_edge_weight(node[1], n) < prev_dict1[n][0]:
            potential_fwd_new = (heuristic(graph, n, goal) - heuristic(graph, n, start)) / 2
            potential_fwd_old = (heuristic(graph, node[1], goal) - heuristic(graph, node[1], start)) / 2
            cost = node[0] - potential_fwd_old + graph.get_edge_weight(node[1], n) + potential_fwd_new
            prev_dict1[n] = (cost, node[1])
            pq.append((cost, n))
            if n in prev_dict2 and prev_dict1[n][0] < optimacy_threshold:
                #print("Made it inside the special if")
                solution_node = n
                optimacy_threshold = prev_dict1[n][0]
    return (solution_node, optimacy_threshold)

def bidirectional_a_star_helper2(graph, node, start, goal, prev_dict1, prev_dict2):
    path = []
    cur = node
    if node != start:
        while cur != start:
            next = prev_dict1[cur][1]
            path.append(next)
            cur = next
    path = path[::-1]
    path.append(node)
    #print("Made it half first half of path")
    if node == goal:
        return path
    cur = node
    while cur != goal:
        next = prev_dict2[cur][1]
        path.append(next)
        cur = next
    #print(path)
    return path

def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def return_your_name():
    """Return your name from this function"""
    return "Pranav Shankar Manjunath"
    raise NotImplementedError


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None
 
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
