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
    return np.linalg.norm(node1.attributes.position - node2.attributes.position, ord=2)

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

    path_list = [goal_node]
    node = path_dict[goal_node]
    while node != None:
        path_list.insert(0, node)
        node = path_dict[node]
    
    return path_list, cost_to_come[goal_node]

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

    Q = PriorityQueue() # min cost_to_come + heuristic_cost_to_go priority queue
    path_dict = {} 
    cost_to_come = {}
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

    path_list = [curr_node]
    node = path_dict[curr_node]
    while node != None:
        path_list.insert(0, node)
        node = path_dict[node]
    
    return path_list, cost_to_come[curr_node]


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
    room_path, _ = layer_astar(G, start_room, goal_room, heuristic)
    # Room to Room planning
    total_path_list = [start_node]
    total_cost = 0.
    for i in range(len(room_path) - 1):
        curr_node = total_path_list.pop()
        path_segment, segment_cost = naive_place_to_room_astar(G, curr_node, room_path[i + 1], heuristic)
        total_path_list.extend(path_segment)
        total_cost += segment_cost
    return total_path_list, total_cost

if __name__ == "__main__":
    path_to_dsg = "./DSGs/uhumans2/backend/dsg.json"
    path_to_dsg = pathlib.Path(path_to_dsg).expanduser().absolute()

    # node data structure: Node<id=node.id.category(node.id.category_id), layer=node.layer>
    # node.id.value: the long number

    G = dsg.DynamicSceneGraph.load(str(path_to_dsg))
    dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)

    # print("Testing Room layer A*")
    # room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
    # room_nodes = list(room_layer.nodes)
    # time1 = time.time()
    # path_list, total_cost = layer_astar(G, room_nodes[1], room_nodes[3], node_dist)
    # time2 = time.time()
    # print(f"Compute time: {time2 - time1} sec")
    # print(path_list)
    # print(total_cost)
    # print()

    print("Testing Place layer A*")
    room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
    room_nodes = list(room_layer.nodes)
    place1 = G.get_node(list(room_nodes[3].children())[0])
    place2 = G.get_node(list(room_nodes[8].children())[10])
    print(place1, place2)
    time1 = time.time()
    path_list, total_cost = layer_astar(G, place1, place2, node_dist)
    time2 = time.time()
    print(f"Compute time: {time2 - time1} sec")
    print(path_list)
    print(total_cost)
    print()

    # print("Testing naive place to room A*")
    # room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
    # room_nodes = list(room_layer.nodes)
    # place1 = G.get_node(list(room_nodes[3].children())[0])
    # place2 = G.get_node(list(room_nodes[8].children())[10])
    # time1 = time.time()
    # path_list, total_cost = naive_place_to_room_astar(G, place1, room_nodes[8], node_dist)
    # time2 = time.time()
    # print(f"Compute time: {time2 - time1} sec")
    # print(path_list)
    # print(total_cost)
    # print()

    print("Testing hierarchical planner")
    room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
    room_nodes = list(room_layer.nodes)
    place1 = G.get_node(list(room_nodes[3].children())[0])
    place2 = G.get_node(list(room_nodes[8].children())[10])
    print(place1, place2)
    time1 = time.time()
    path_list, total_cost = hierarchical_planner(G, place1, place2, node_dist)
    time2 = time.time()
    print(f"Compute time: {time2 - time1} sec")
    print(path_list)
    print(total_cost)
    print()