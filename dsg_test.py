# %%
import spark_dsg as dsg
import pathlib

# %%
path_to_dsg = "./DSGs/uhumans2/backend/dsg.json"
path_to_dsg = pathlib.Path(path_to_dsg).expanduser().absolute()


# %%
G = dsg.DynamicSceneGraph.load(str(path_to_dsg))

# %%
layer_settings = {}
for layer_idx, layer in enumerate(G.layers):
    if layer_idx == 1:
            layer_settings[layer.id] = {'marker_size': 2, 'annotation_size': 25, 'include_text': True}
    else:
        layer_settings[layer.id] = {'marker_size': 6+layer_idx, 'annotation_size': 25, 'include_text': True}
fig = dsg.plot_scene_graph(G, title="Test DSG", figure_path="plots/test_dsg.html", layer_settings=layer_settings)
if fig is not None:
    fig.show()


# %% [markdown]
# ## Node Access and Node Attributes


# %%
def print_node_if_exists(node_id):
    """Grab node attributes and display if the graph contains the node."""
    test_str = "yes" if G.has_node(node_id.value) else "no"
    print(f"Graph contains node {node_id}: {test_str}")

    if not G.has_node(node_id.value):
        return

    node = G.get_node(node_id.value)
    print(f"Node {node_id} attributes:\n{node.attributes}")


# object node
print("")
print_node_if_exists(dsg.NodeSymbol("O", 4))

# places node
print("")
print_node_if_exists(dsg.NodeSymbol("p", 135))

# room node
print("")
print_node_if_exists(dsg.NodeSymbol("R", 19))


# agent node
print("")
print_node_if_exists(dsg.NodeSymbol("a", 5))


# %% [markdown]
# ## Layer Iteration and Access

# %%
print("Layers:")
for layer in G.layers:
    print(f"  - {layer.id}")
print("")

# %%
room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
room_node_strs = [f"{x.id}" for x in room_layer.nodes]
print(f"Rooms: {room_node_strs}")

room_edge_strs = [f"  - {x}" for x in room_layer.edges]
print("Room Edges:")
print("\n".join(room_edge_strs))
print("")


# %% [markdown]
# ## Interlayer Edge Access

# %%
layer_edge_counts = {}
places = set() # MP test
places_in_multi_rooms = set() # MP test
for edge in G.interlayer_edges:

    source_layer = G.get_node(edge.source).layer
    target_layer = G.get_node(edge.target).layer

    # MP test
    if source_layer == 4:
        target_id = G.get_node(edge.target).id
        if id in places:
            places_in_multi_rooms.add(target_id)
        places.add(target_id)
    # MP test

    if source_layer not in layer_edge_counts:
        layer_edge_counts[source_layer] = {}
    if target_layer not in layer_edge_counts[source_layer]:
        layer_edge_counts[source_layer][target_layer] = 0

    layer_edge_counts[source_layer][target_layer] += 1

print("Interlayer Edges:")
for source_layer in layer_edge_counts:
    print(f"  - {source_layer} -> {layer_edge_counts[source_layer]}")
print(f"Places in Multiple Rooms: {places_in_multi_rooms}") # MP test
print(f"Total places count: {len(places)}") # MP test
print("")


# %% [markdown]
# ## Entire Graph Access

# %%
node_type_counts = {}
for node in G.nodes:
    if node.id.category not in node_type_counts:
        node_type_counts[node.id.category] = 0

    node_type_counts[node.id.category] += 1

print("Node Types:")
for category, count in node_type_counts.items():
    print(f"  - {category}: {count}")


edge_counts = {}
for edge in G.edges:
    source_layer = G.get_node(edge.source).layer
    target_layer = G.get_node(edge.target).layer
    if source_layer not in edge_counts:
        edge_counts[source_layer] = {}
    if target_layer not in edge_counts[source_layer]:
        edge_counts[source_layer][target_layer] = 0

    edge_counts[source_layer][target_layer] += 1

print("All Edges:")
for source_layer in edge_counts:
    print(f"  - {source_layer} -> {edge_counts[source_layer]}")
print("")


# %% [markdown]
# ## Bounding boxes from nodes in lower layers

# %%
dsg.add_bounding_boxes_to_layer(G, dsg.DsgLayers.ROOMS)
print("Room bounding boxes:")
for node in G.get_layer(dsg.DsgLayers.ROOMS).nodes:
    print(f"  - {node.id}: {node.attributes.bounding_box}")