import networkx as nx
import random
import time


def generate_large_synthetic_graph(
        num_nodes=100000, avg_degree=5, graph_type="mixed", seed=42):
    """
    Generates a complex synthetic graph dataset.

    Parameters:
        num_nodes (int): Number of nodes in the graph.
        avg_degree (int): Average number of edges per node.
        graph_type (str): Type of graph to generate. Options:
                          - "scale_free" (preferential attachment)
                          - "small_world" (Watts-Strogatz model)
                          - "random" (Erdős-Rényi)
                          - "mixed" (combines multiple graphs)
        seed (int): Random seed for reproducibility.

    Returns:
        networkx.Graph: The generated synthetic graph.
    """
    random.seed(seed)
    nx.seed = seed

    print(f"Generating {graph_type} graph with {num_nodes} nodes...")

    if graph_type == "scale_free":
        G = nx.barabasi_albert_graph(num_nodes, avg_degree)
    elif graph_type == "small_world":
        G = nx.watts_strogatz_graph(num_nodes, avg_degree, 0.1)
    elif graph_type == "random":
        G = nx.erdos_renyi_graph(num_nodes, avg_degree / num_nodes)
    elif graph_type == "mixed":
        G1 = nx.barabasi_albert_graph(num_nodes // 2, avg_degree)
        G2 = nx.watts_strogatz_graph(num_nodes // 2, avg_degree, 0.2)
        G = nx.compose(G1, G2)
        for _ in range(avg_degree):  # Add random edges between components
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u != v:
                G.add_edge(u, v)
    else:
        raise ValueError(
            "Invalid graph_type. Choose from 'scale_free', 'small_world', 'random', or 'mixed'.")

    print(
        f"Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def save_graph_to_metis(G, file_name="synthetic_large_graph.graph"):
    """
    Saves a NetworkX graph to METIS format for PRRP processing.

    Parameters:
        G (networkx.Graph): The graph to save.
        file_name (str): Output filename.

    Returns:
        None
    """
    print(f"Saving graph to {file_name} in METIS format...")
    with open(file_name, "w") as f:
        num_nodes = G.number_of_nodes()
        # METIS expects undirected edges counted once
        num_edges = G.number_of_edges() // 2
        f.write(f"{num_nodes} {num_edges}\n")
        for node in range(num_nodes):
            neighbors = " ".join(str(n + 1)
                                 for n in G.neighbors(node))  # 1-based indexing
            f.write(neighbors + "\n")
    print("Graph successfully saved.")


if __name__ == "__main__":
    start_time = time.time()
    graph = generate_large_synthetic_graph(
        num_nodes=100000, avg_degree=5, graph_type="mixed")
    save_graph_to_metis(graph, "data/sample/synthetic_large_graph_100k.graph")
    print(
        f"Graph generation completed in {time.time() - start_time:.2f} seconds.")
