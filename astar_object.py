from astar import *

def nav_to_object(G, start_node, goal_object, method, heuristics):
    # convert input from node.id.value to node
    if isinstance(start_node, int):
        start_node = G.get_node(start_node)
    
    if start_node.layer != 3:
        raise Exception("Start node is not a place")
    
    object_layer = G.get_layer(dsg.DsgLayers.OBJECTS)
    object_nodes = list(object_layer.nodes)
    best_place = None
    best_dist = np.inf
    for obj in object_nodes:
        if goal_object == obj.attributes.name:
            obj_parent_value = obj.get_parent()
            if obj_parent_value is None:
                continue
            place_node = G.get_node(obj_parent_value)
            dist = node_dist(start_node, place_node)
            if dist < best_dist:
                best_place = place_node
                best_dist = dist
    if best_place is None:
        raise Exception("No place found that corresponds to the desired object")
    if method == layer_astar:
        return get_info(*method(G, start_node, best_place, heuristics))
    return method(G, start_node, best_place, heuristics)


path_to_dsg = "./DSGs/uhumans2/backend/dsg.json"
path_to_dsg = pathlib.Path(path_to_dsg).expanduser().absolute()

# node data structure: Node<id=node.id.category(node.id.category_id), layer=node.layer>
# node.id.value: the long number

G = dsg.DynamicSceneGraph.load(str(path_to_dsg))
dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)

object_layer = G.get_layer(dsg.DsgLayers.OBJECTS)
object_nodes = list(object_layer.nodes)
for obj in object_nodes:
    print(obj.attributes.name)


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

# print("Testing Place layer A*")
# room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
# room_nodes = list(room_layer.nodes)
# place1 = G.get_node(list(room_nodes[3].children())[0])
# place2 = G.get_node(list(room_nodes[8].children())[10])
# print(place1, place2)
# time1 = time.time()
# path_list, total_cost = get_info(*layer_astar(G, place1, place2, node_dist))
# time2 = time.time()
# print(f"Compute time: {time2 - time1} sec")
# print(path_list)
# print(total_cost)
# print()

# print("Testing naive place to room A*")
# room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
# room_nodes = list(room_layer.nodes)
# place1 = G.get_node(list(room_nodes[3].children())[0])
# place2 = G.get_node(list(room_nodes[8].children())[10])
# time1 = time.time()
# path_list, total_cost = get_info(*naive_place_to_room_astar(G, place1, room_nodes[8], node_dist))
# time2 = time.time()
# print(f"Compute time: {time2 - time1} sec")
# print(path_list)
# print(total_cost)
# print()

# print("Testing hierarchical planner")
# room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
# room_nodes = list(room_layer.nodes)
# place1 = G.get_node(list(room_nodes[3].children())[0])
# place2 = G.get_node(list(room_nodes[8].children())[10])
# print(place1, place2)
# time1 = time.time()
# path_list, total_cost = hierarchical_planner(G, place1, place2, node_dist)
# time2 = time.time()
# print(f"Compute time: {time2 - time1} sec")
# print(path_list)
# print(total_cost)
# print()

print("Testing object navigation (layer A*)")
objects = ["fan_papers_keyboard_laptop_mouse", "trashcan", "chair", "plant", "painting", "couch"]
print(f"Choose object from {objects}")
obj = "chair"

room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
room_nodes = list(room_layer.nodes)
place1 = G.get_node(list(room_nodes[8].children())[0])

time1 = time.time()
path_list, total_cost = nav_to_object(G, place1, obj, layer_astar, node_dist)
time2 = time.time()
print(f"Compute time: {time2 - time1} sec")
print(path_list)
print(total_cost)
print()
print("Testing object navigation (hierarchical_planner)")
time1 = time.time()
path_list, total_cost = nav_to_object(G, place1, obj, hierarchical_planner, node_dist)
time2 = time.time()
print(f"Compute time: {time2 - time1} sec")
print(path_list)
print(total_cost)
print()