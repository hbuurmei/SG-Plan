from astar import *
import matplotlib.pyplot as plt


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

# print("Testing object navigation (layer A*)")
objects = ["fan_papers_keyboard_laptop_mouse", "trashcan", "chair", "plant", "painting", "couch"]
print(f"Choose object from {objects}")
target_object = "couch"

room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
room_nodes = list(room_layer.nodes)
place1 = G.get_node(list(room_nodes[8].children())[0])

print("Testing object navigation (hierarchical_planner)")
time1 = time.time()
path_list, total_cost = nav_to_object(G, place1, target_object, hierarchical_planner, node_dist)
time2 = time.time()
print(f"Compute time: {time2 - time1} sec")
print(total_cost)
print()

print("--- Plotting paths ---")
start_pos = place1.attributes.position
places_layer = G.get_layer(dsg.DsgLayers.PLACES)
object_layer = G.get_layer(dsg.DsgLayers.OBJECTS)
object_nodes = list(object_layer.nodes)
plt.figure(figsize=(8, 8))

colors = [
    "#f4c2c2",  # Light Pink
    "#ace1af",  # Soft Celadon Green
    "#add8e6",  # Light Blue
    "#bdb76b",  # Dark Khaki
    "#ffe4e1",  # Misty Rose
    "#77dd77",  # Pastel Green
    "#fdd5b1",  # Light Apricot
    "#9f79ee",  # Medium Purple
    "#cfcfc4"   # Pastel Gray
]
node_to_room_color = {}
for i, room in enumerate(room_layer.nodes):
    room_color = colors[i]
    plt.plot(room.attributes.position[0], room.attributes.position[1], "o", color=room_color, markersize=5, alpha=0.8)
    plt.text(room.attributes.position[0]-0.5, room.attributes.position[1]+0.5, f"$R_{i}$", color=colors[i], alpha=1.0, fontsize=11)  # plot room centroid
    for node_value in room.children():
        node = G.get_node(node_value)
        plt.plot(node.attributes.position[0], node.attributes.position[1], "o", color=room_color, markersize=2, alpha=0.25)
        node_to_room_color[node_value] = room_color
for edge in places_layer.edges:
    edge_color = node_to_room_color.get(edge.source)
    start = G.get_node(edge.source).attributes.position
    end = G.get_node(edge.target).attributes.position
    plt.plot([start[0], end[0]], [start[1], end[1]], color=edge_color, alpha=0.25, linewidth=0.3)

for node in path_list:
    # Plot the nodes in the path
    plt.plot(node.attributes.position[0], node.attributes.position[1], "o", color="green", markersize=5)
for i in range(len(path_list) - 1):
    # Plot the edges in the path
    start = path_list[i].attributes.position
    end = path_list[i + 1].attributes.position
    plt.plot([start[0], end[0]], [start[1], end[1]], color="green", linewidth=2, label="Hierarchical A* Path" if i == 0 else "")

plt.plot(start_pos[0], start_pos[1], 'x', color='black', markersize=8, label='Start point')
for obj in object_nodes:
        if target_object == obj.attributes.name:
            obj_parent_value = obj.get_parent()
            if obj_parent_value is None:
                continue
            place_node = G.get_node(obj_parent_value)
            plt.plot(place_node.attributes.position[0], place_node.attributes.position[1], 'x', color='red', markersize=8)            

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Hierarchical A* Path to Object")
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')  # ensure physical aspect ratio is maintained
plt.savefig("plots/astar_object.png")
