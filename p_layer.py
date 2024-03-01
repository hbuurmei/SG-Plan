import spark_dsg as dsg
import pathlib

# %%
path_to_dsg = "./DSGs/uhumans2/backend/dsg.json"
path_to_dsg = pathlib.Path(path_to_dsg).expanduser().absolute()


# %%
G = dsg.DynamicSceneGraph.load(str(path_to_dsg))

place_layer = G.get_layer(dsg.DsgLayers.PLACES)
print(f"Number of places: {place_layer.num_nodes()}")
print(f"Number of edges between places: {place_layer.num_edges()}")

for edge in place_layer.edges:
    source = G.get_node(edge.source)
    target = G.get_node(edge.target)
    print(source, target)
    print(source.id, target.id)
    print(source.attributes.position, target.attributes.position)
    print(edge.info.weight) # what is this?
    break