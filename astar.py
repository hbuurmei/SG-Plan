import spark_dsg as dsg
import pathlib
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Any
from queue import PriorityQueue


@dataclass(order=True)
class PrioritizedNode:
    # data wrapper class so that the nodes don't get compared
    node: Any=field(compare=False)
    priority: float

def bad_heuristic(node1, node2):
    # mimics djikstra
    return 0. 

def node_dist(node1, node2):
    return np.linalg.norm(node1.attributes.position[:2] - node2.attributes.position[:2], ord=2)

def get_info(goal_node, path_dict, cost_to_come):
    # returns usable information from dictionary
    path_list = [goal_node]
    node = path_dict[goal_node]
    while node != None:
        path_list.insert(0, node)
        node = path_dict[node]
    
    return path_list, cost_to_come[goal_node]

def layer_astar(G, start_node, goal_node, heuristic):
    # convert input from node.id.value to node
    if isinstance(start_node, int):
        start_node = G.get_node(start_node)
    if isinstance(goal_node, int):
        goal_node = G.get_node(goal_node)
    
    if start_node.layer != goal_node.layer:
        raise Exception("Start and goal not on the same layer")

    Q = PriorityQueue() # min cost_to_come + heuristic_cost_to_go priority queue
    path_dict = {} 
    cost_to_come = {}
    Q.put(PrioritizedNode(start_node, 0.))
    path_dict[start_node] = None
    cost_to_come[start_node] = 0.

    while not Q.empty():
        curr_node = Q.get().node
        if curr_node == goal_node:
            break
        for node_value in curr_node.siblings():
            neighbor_node = G.get_node(node_value) # convert the long number to node
            cost = cost_to_come[curr_node] + node_dist(curr_node, neighbor_node)
            if neighbor_node not in cost_to_come or cost < cost_to_come[neighbor_node]:
                cost_to_come[neighbor_node] = cost
                priority = cost + heuristic(neighbor_node, goal_node)
                Q.put(PrioritizedNode(neighbor_node, priority))
                path_dict[neighbor_node] = curr_node
    
    return goal_node, path_dict, cost_to_come

def naive_place_to_room_astar(G, start_node, goal_room, heuristic):
    # convert input from node.id.value to node
    if isinstance(start_node, int):
        start_node = G.get_node(start_node)
    if isinstance(goal_room, int):
        goal_room = G.get_node(goal_room)
    
    if start_node.layer != 3:
        raise Exception("Start node is not a place")
    if goal_room.layer != 4:
        raise Exception("Goal node is not a room")

    path_dict={}
    cost_to_come={}

    Q = PriorityQueue() # min cost_to_come + heuristic_cost_to_go priority queue
    Q.put(PrioritizedNode(start_node, 0.))
    path_dict[start_node] = None
    cost_to_come[start_node] = 0.
    goal_room_place_values = goal_room.children()

    while not Q.empty():
        curr_node = Q.get().node
        if curr_node.id.value in goal_room_place_values:
            break
        for node_value in curr_node.siblings():
            neighbor_node = G.get_node(node_value) # convert the long number to node
            cost = cost_to_come[curr_node] + node_dist(curr_node, neighbor_node)
            if neighbor_node not in cost_to_come or cost < cost_to_come[neighbor_node]:
                cost_to_come[neighbor_node] = cost
                priority = cost + heuristic(neighbor_node, goal_room)
                Q.put(PrioritizedNode(neighbor_node, priority))
                path_dict[neighbor_node] = curr_node
    
    return curr_node, path_dict, cost_to_come

def closest_place_to_room_astar(G, start_node, goal_room, heuristic): # bad
    # convert input from node.id.value to node
    if isinstance(start_node, int):
        start_node = G.get_node(start_node)
    if isinstance(goal_room, int):
        goal_room = G.get_node(goal_room)
    
    if start_node.layer != 3:
        raise Exception("Start node is not a place")
    if goal_room.layer != 4:
        raise Exception("Goal node is not a room")

    # get closest_goal_room_node
    idx = 0
    room_node_pos = np.zeros((len(goal_room.children()), 3), dtype=float)
    values = np.zeros(len(goal_room.children()), dtype=int)
    for value in goal_room.children():
        room_node_pos[idx, :] = G.get_node(value).attributes.position
        values[idx] = value
        idx += 1
    room_node_dists = np.linalg.norm(room_node_pos, ord=2, axis=1)
    closest_room_node = G.get_node(values[np.argmin(room_node_dists)])
    return layer_astar(G, start_node, closest_room_node, heuristic)


def hierarchical_planner(G, start_node, goal_node, heuristic):
    # convert input from node.id.value to node
    if isinstance(start_node, int):
        start_node = G.get_node(start_node)
    if isinstance(goal_node, int):
        goal_node = G.get_node(goal_node)
    # make sure the nodes are places
    if start_node.layer != 3:
        raise Exception("Start node is not a place")
    if goal_node.layer != 3:
        raise Exception("Goal node is not a place")

    # Room level planning
    start_room = G.get_node(start_node.get_parent())
    goal_room = G.get_node(goal_node.get_parent())
    room_path, _ = get_info(*layer_astar(G, start_room, goal_room, heuristic))
    # Room to Room planning
    total_path_list = [start_node]
    total_cost = 0.
    for i in range(len(room_path)):
        curr_node = total_path_list.pop()
        if i == len(room_path) - 1:
            path_segment, segment_cost = get_info(*layer_astar(G, curr_node, goal_node, heuristic))
        else:
            path_segment, segment_cost = get_info(*naive_place_to_room_astar(G, curr_node, room_path[i + 1], heuristic))
        total_path_list.extend(path_segment)
        total_cost += segment_cost
    return total_path_list, total_cost
