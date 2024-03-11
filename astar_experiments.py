import time
import pathlib
import spark_dsg as dsg
import matplotlib.pyplot as plt
from astar import get_info, node_dist, layer_astar, hierarchical_planner  # TODO: change to single run_astar function


path_to_dsg = "./DSGs/uhumans2/backend/dsg.json"
path_to_dsg = pathlib.Path(path_to_dsg).expanduser().absolute()

# node data structure: Node<id=node.id.category(node.id.category_id), layer=node.layer>
# node.id.value: the long number

G = dsg.DynamicSceneGraph.load(str(path_to_dsg))
dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)

print("--- Testing place layer A* ---")
room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
room_nodes = list(room_layer.nodes)
place1 = G.get_node(list(room_nodes[3].children())[0])
place2 = G.get_node(list(room_nodes[8].children())[10])
time1 = time.time()
path_list_layer, total_cost = get_info(*layer_astar(G, place1, place2, node_dist))
time2 = time.time()
print(f"Start: {place1}")
print(f"End: {place2}")
print(f"Compute time: {(time2 - time1):.5f} sec")
print(f"Cost: {total_cost:.3f} \n")

print("--- Testing hierarchical planner A* ---")
room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
room_nodes = list(room_layer.nodes)
place1 = G.get_node(list(room_nodes[3].children())[0])
place2 = G.get_node(list(room_nodes[8].children())[10])
time1 = time.time()
path_list_hierarchical, total_cost = hierarchical_planner(G, place1, place2, node_dist)
time2 = time.time()
print(f"Start: {place1}")
print(f"End: {place2}")
print(f"Compute time: {(time2 - time1):.5f} sec")
print(f"Cost: {total_cost:.3f} \n")

# Plot the two paths on top of the DSG in 2D
print("--- Plotting paths ---")
start_pos = place1.attributes.position
end_pos = place2.attributes.position
places_layer = G.get_layer(dsg.DsgLayers.PLACES)
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

for node in path_list_layer:
    # Plot the nodes in the path
    plt.plot(node.attributes.position[0], node.attributes.position[1], "o", color="red", markersize=5)
for i in range(len(path_list_layer) - 1):
    # Plot the edges in the path
    start = path_list_layer[i].attributes.position
    end = path_list_layer[i + 1].attributes.position
    plt.plot([start[0], end[0]], [start[1], end[1]], color="red", linewidth=2, label="Layer A* Path" if i == 0 else "")

for node in path_list_hierarchical:
    # Plot the nodes in the path
    plt.plot(node.attributes.position[0], node.attributes.position[1], "o", color="green", markersize=5)
for i in range(len(path_list_hierarchical) - 1):
    # Plot the edges in the path
    start = path_list_hierarchical[i].attributes.position
    end = path_list_hierarchical[i + 1].attributes.position
    plt.plot([start[0], end[0]], [start[1], end[1]], color="green", linewidth=2, label="Hierarchical A* Path" if i == 0 else "")

plt.plot(start_pos[0], start_pos[1], 'x', color='black', markersize=8, label='Start point')
plt.plot(end_pos[0], end_pos[1], 'D', color='black', markersize=5, label='End point')

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Layer A* vs. Hierarchical A* Path Comparison")
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')  # ensure physical aspect ratio is maintained
plt.savefig("plots/astar_experiment.png")