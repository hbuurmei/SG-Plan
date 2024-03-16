from astar import *
import pickle
import matplotlib.pyplot as plt


def save_data(N):
    path_to_dsg = "./DSGs/uhumans2/backend/dsg.json"
    path_to_dsg = pathlib.Path(path_to_dsg).expanduser().absolute()

    G = dsg.DynamicSceneGraph.load(str(path_to_dsg))
    room_layer = G.get_layer(dsg.DsgLayers.ROOMS)
    room_nodes = list(room_layer.nodes)
    num_rooms = len(room_nodes)
    room_sizes = np.zeros(num_rooms, dtype=int)
    for idx in range(num_rooms):
        room_sizes[idx] = len(room_nodes[idx].children())

    in_room_place_layer = {'times':[], 'costs':[]}
    in_room_hierarchical = {'times':[], 'costs':[]}
    cross_room_place_layer = {'times':[], 'costs':[]}
    cross_room_hierarchical = {'times':[], 'costs':[]}

    # randomly select rooms and nodes from rooms to plan paths between them
    # record resluts in different dicts to distinguish between in_room and cross_room planned paths
    room_indices = np.random.choice(np.arange(num_rooms), (N, 2))

    t0 = time.time()
    for idx in range(N):
        room_idx1, room_idx2 = room_indices[idx]
        if room_idx1 == room_idx2:
            node_idx1, node_idx2 = np.random.choice(np.arange(room_sizes[room_idx1]), 2, replace=False)
        else:
            node_idx1 = np.random.choice(np.arange(room_sizes[room_idx1]))
            node_idx2 = np.random.choice(np.arange(room_sizes[room_idx2]))
        place1 = G.get_node(list(room_nodes[room_idx1].children())[node_idx1])
        place2 = G.get_node(list(room_nodes[room_idx2].children())[node_idx2])
        t1 = time.time()
        _, layer_cost = get_info(*layer_astar(G, place1, place2, node_dist))
        t2 = time.time()
        _, hier_cost = hierarchical_planner(G, place1, place2, node_dist)
        t3 = time.time()
        if room_idx1 == room_idx2:
            in_room_place_layer['times'].append(t2 - t1)
            in_room_place_layer['costs'].append(layer_cost)
            in_room_hierarchical['times'].append(t3 - t2)
            in_room_hierarchical['costs'].append(hier_cost)
        else:
            cross_room_place_layer['times'].append(t2 - t1)
            cross_room_place_layer['costs'].append(layer_cost)
            cross_room_hierarchical['times'].append(t3 - t2)
            cross_room_hierarchical['costs'].append(hier_cost)

    tf = time.time()
    print(f"Total compute time: {tf - t0}")

    data = {"in_room_place_layer": in_room_place_layer, 
            "in_room_hierarchical": in_room_hierarchical, 
            "cross_room_place_layer": cross_room_place_layer, 
            "cross_room_hierarchical": cross_room_hierarchical}
    with open("result_dicts.pkl", "wb") as outfile:
        pickle.dump(data, outfile)


if __name__ == "__main__":
    np.random.seed(0) # not working???
    N = 1000
    # save_data(N) # comment out if re-compute not needed, takes a while

    with open("result_dicts.pkl", "rb") as f:
        data = pickle.load(f)
    
    in_room_place_layer = data["in_room_place_layer"]
    in_room_hierarchical = data["in_room_hierarchical"]
    cross_room_place_layer = data["cross_room_place_layer"]
    cross_room_hierarchical = data["cross_room_hierarchical"]

    print(f"--- In_room results ({len(in_room_hierarchical['times'])}) --- ")
    optimal_costs_in = np.array(in_room_place_layer['costs'])
    hier_costs_in = np.array(in_room_hierarchical['costs'])
    time_ratios_in = np.array(in_room_hierarchical['times']) / np.array(in_room_place_layer['times'])
    print(f"Layer time = {np.mean(in_room_place_layer['times'])} +- {np.std(in_room_place_layer['times'])}")
    print(f"Layer cost = {np.mean(in_room_place_layer['costs'])} +- {np.std(in_room_place_layer['costs'])}")
    print(f"Hierarchical time = {np.mean(in_room_hierarchical['times'])} +- {np.std(in_room_hierarchical['times'])}")
    print(f"Hierarchical cost = {np.mean(in_room_hierarchical['costs'])} +- {np.std(in_room_hierarchical['costs'])}")
    print(f"Time ratio = {np.mean(time_ratios_in)}")
    print(f"Cost ratio = {np.mean(hier_costs_in / optimal_costs_in)}")
    print()

    print(f"--- Cross_room results ({len(cross_room_hierarchical['times'])}) ---")
    optimal_costs_cross = np.array(cross_room_place_layer['costs'])
    hier_costs_cross = np.array(cross_room_hierarchical['costs'])
    time_ratios_cross = np.array(cross_room_hierarchical['times']) / np.array(cross_room_place_layer['times'])
    print(f"Layer time = {np.mean(cross_room_place_layer['times'])} +- {np.std(cross_room_place_layer['times'])}")
    print(f"Layer cost = {np.mean(cross_room_place_layer['costs'])} +- {np.std(cross_room_place_layer['costs'])}")
    print(f"Hierarchical time = {np.mean(cross_room_hierarchical['times'])} +- {np.std(cross_room_hierarchical['times'])}")
    print(f"Hierarchical cost = {np.mean(cross_room_hierarchical['costs'])} +- {np.std(cross_room_hierarchical['costs'])}")
    print(f"Time ratio = {np.mean(time_ratios_cross)}")
    print(f"Cost ratio = {np.mean(hier_costs_cross / optimal_costs_cross)}")
    print()

    print(f"--- Combined results ({optimal_costs_in.size + optimal_costs_cross.size}) ---")
    print(f"Time ratio = {np.mean(np.concatenate((time_ratios_in, time_ratios_cross)))}")
    print(f"Cost ratio = {np.mean(np.concatenate((hier_costs_in, hier_costs_cross)) / np.concatenate((optimal_costs_in, optimal_costs_cross)))}")
    print(f"Time ratio stdev = {np.std(np.concatenate((time_ratios_in, time_ratios_cross)))}")
    print(f"Cost ratio stdev = {np.std(np.concatenate((hier_costs_in, hier_costs_cross)) / np.concatenate((optimal_costs_in, optimal_costs_cross)))}")

    alpha = 0.5
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(optimal_costs_in, hier_costs_in / optimal_costs_in, marker='.', alpha=alpha, label="in room")
    ax.scatter(optimal_costs_cross, hier_costs_cross / optimal_costs_cross, marker='.', alpha=alpha, label="cross room")
    # ax.set_yscale("log")
    ax.set_xlabel("Optimal path length [m]")
    ax.set_ylabel("Path length ratio")
    ax.legend()
    # ax.set_title("Hierarchical Planner Cost Suboptimality wrt. True Shortest Path Length")
    fig.savefig("plots/costs_ratio.png")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(optimal_costs_in, time_ratios_in, marker='.', alpha=alpha,  label="in room")
    ax.scatter(optimal_costs_cross, time_ratios_cross, marker='.', alpha=alpha, label="cross room")
    ax.set_yscale("log")
    ax.set_xlabel("Optimal path length [m]")
    ax.set_ylabel("Compute time ratio")
    ax.legend()
    # ax.set_title("Hierarchical Planner Compute Efficiency wrt. True Shortest Path Length")
    fig.savefig("plots/times_ratio.png")

    fig, ax = plt.subplots(figsize=(8, 8))
    # ax.scatter(optimal_costs_in, hier_costs_in / optimal_costs_in, marker='.', alpha=alpha, label="in room")
    ax.scatter(optimal_costs_cross, hier_costs_cross / optimal_costs_cross, marker='.', alpha=alpha, label="Layer A*")
    # ax.scatter(optimal_costs_cross, hier_costs_cross, marker='.', alpha=alpha, label="Hierarchical A*")
    # ax.set_yscale("log")
    ax.set_xlabel("Optimal path length [m]")
    ax.set_ylabel("Path length ratio [-]")
    # ax.legend()
    # ax.set_title("Hierarchical Planner Cost Suboptimality wrt. True Shortest Path Length")
    fig.savefig("plots/costs.png")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(optimal_costs_cross, np.array(cross_room_hierarchical['times'])/np.array(cross_room_place_layer['times']), marker='.', alpha=alpha)
    # ax.scatter(optimal_costs_cross, cross_room_hierarchical['times'], marker='.', alpha=alpha, label="Hierarchical A*")
    ax.set_yscale("log")
    ax.set_xlabel("Optimal path length [m]")
    ax.set_ylabel("Compute time ratio [-]")
    # ax.legend()
    # ax.set_title("Hierarchical Planner Compute Efficiency wrt. True Shortest Path Length")
    fig.savefig("plots/times.png")
