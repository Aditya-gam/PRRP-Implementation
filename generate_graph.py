import networkx as nx
import random
import time
import os


def generate_large_synthetic_graph(
        num_nodes=100000, avg_degree=5, graph_type="mixed", seed=42):
    """
    Generates a large synthetic graph dataset compatible with PRRP.

    The function supports different graph models:
      - "scale_free": Barabási-Albert model (preferential attachment)
      - "small_world": Watts-Strogatz model (small-world properties)
      - "random": Erdős-Rényi model (uniform random connections)
      - "mixed": Combination of scale-free and small-world models, fully connected

    Parameters:
        num_nodes (int): Number of nodes in the graph.
        avg_degree (int): Average degree (number of edges per node).
        graph_type (str): Type of graph to generate. Options: "scale_free", "small_world", "random", "mixed".
        seed (int): Random seed for reproducibility.

    Returns:
        networkx.Graph: A fully connected, 1-based indexed undirected graph.
    """
    random.seed(seed)  # Ensuring reproducibility

    print(f"Generating a {graph_type} graph with {num_nodes} nodes...")

    # Initialize graph
    if graph_type == "scale_free":
        G = nx.barabasi_albert_graph(num_nodes, avg_degree, seed=seed)
    elif graph_type == "small_world":
        G = nx.watts_strogatz_graph(num_nodes, avg_degree, 0.1, seed=seed)
    elif graph_type == "random":
        G = nx.erdos_renyi_graph(num_nodes, avg_degree / num_nodes, seed=seed)
    elif graph_type == "mixed":
        half_nodes = num_nodes // 2

        # Generate two different subgraphs
        G1 = nx.barabasi_albert_graph(half_nodes, avg_degree, seed=seed)
        G2 = nx.watts_strogatz_graph(half_nodes, avg_degree, 0.2, seed=seed)

        # Relabel G2 so that its node indices don’t overlap with G1
        mapping = {old: old + half_nodes for old in G2.nodes()}
        G2 = nx.relabel_nodes(G2, mapping)

        # Merge graphs
        G = nx.compose(G1, G2)

        # Ensure full connectivity by adding bridging edges
        for _ in range(avg_degree):
            u = random.choice(list(G1.nodes()))
            v = random.choice(list(G2.nodes()))
            G.add_edge(u, v)
    else:
        raise ValueError(
            "Invalid graph_type. Choose from 'scale_free', 'small_world', 'random', or 'mixed'.")

    # Ensure 1-based node indexing (METIS format expects 1-based indexing)
    mapping = {node: node + 1 for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    print(
        f"Generated graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def save_graph_to_metis(G, file_name="synthetic_large_graph.graph"):
    """
    Saves a NetworkX graph in METIS format for PRRP compatibility.

    METIS expects:
      - 1-based node indexing.
      - The first line contains: `num_nodes num_edges`
      - Each subsequent line lists the neighbors of a node.

    Parameters:
        G (networkx.Graph): The input undirected graph.
        file_name (str): Path to save the METIS formatted graph.

    Returns:
        None
    """
    print(f"Saving graph to {file_name} in METIS format...")

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "w") as f:
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges() // 2  # Undirected edges are counted once

        f.write(f"{num_nodes} {num_edges}\n")

        for node in sorted(G.nodes()):
            neighbors = " ".join(str(n) for n in sorted(G.neighbors(node)))
            f.write(neighbors + "\n")

    print(f"Graph successfully saved to {file_name}.")


if __name__ == "__main__":
    start_time = time.time()

    # Generate and save a large synthetic graph
    graph = generate_large_synthetic_graph(
        num_nodes=100000, avg_degree=5, graph_type="mixed")
    save_graph_to_metis(graph, "data/sample/synthetic_large_graph_100k.graph")

    print(
        f"Graph generation completed in {time.time() - start_time:.2f} seconds.")
