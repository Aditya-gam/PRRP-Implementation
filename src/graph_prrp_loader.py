import networkx as nx
import logging
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

def load_metis_graph(file_path: str) -> (nx.Graph, dict):
    """
    Parses a METIS graph file and loads it into an adjacency list.

    Parameters:
        file_path: Path to the METIS graph file

    Returns:
        nx.Graph: NetworkX graph object
        dict: Adjacency list
    """
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None, None
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None, None

    try:
        header = lines[0].strip().split()
        num_nodes, num_edges = int(header[0]), int(header[1])
    except Exception as e:
        logger.error(f"Error parsing header in file {file_path}: {e}")
        return None, None

    adjacency_list = {}

    try:
        for node_id, line in enumerate(lines[1:], start=1):
            neighbors = list(map(lambda x: int(x) - 1, line.strip().split()))
            adjacency_list[node_id - 1] = neighbors  # Convert to 0-based index
    except Exception as e:
        logger.error(f"Error parsing adjacency list in file {file_path}: {e}")
        return None, None

    G = nx.Graph()
    try:
        for node, neighbors in adjacency_list.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
    except Exception as e:
        logger.error(f"Error creating NetworkX graph: {e}")
        return None, None

    return G, adjacency_list

def preprocess_graph(G: nx.Graph, remove_self_loops=True) -> nx.Graph:
    """
    Preprocesses the graph by removing self-loops and ensuring it is undirected.

    Parameters:
        G: NetworkX graph object
        remove_self_loops: Boolean flag to remove self-loops

    Returns:
        nx.Graph: Preprocessed graph
    """
    try:
        if remove_self_loops:
            G.remove_edges_from(nx.selfloop_edges(G))
            logger.debug("Removed self-loops.")

        G = G.to_undirected()
        logger.debug("Converted to undirected graph.")
    except Exception as e:
        logger.error(f"Error preprocessing graph: {e}")
        return None

    return G

def convert_to_adjacency_list(G: nx.Graph) -> dict:
    """
    Converts the graph to an adjacency list format.

    Parameters:
        G: NetworkX graph object

    Returns:
        dict: Dictionary representing adjacency list
    """
    try:
        return {node: list(G.neighbors(node)) for node in G.nodes()}
    except Exception as e:
        logger.error(f"Error converting to adjacency list: {e}")
        return None

def convert_to_edge_list(G: nx.Graph) -> list:
    """
    Converts the graph to an edge list format.

    Parameters:
        G: NetworkX graph object

    Returns:
        list: List of edges
    """
    try:
        return list(G.edges())
    except Exception as e:
        logger.error(f"Error converting to edge list: {e}")
        return None

def convert_to_adjacency_matrix(G: nx.Graph) -> np.ndarray:
    """
    Converts the graph to an adjacency matrix.

    Parameters:
        G: NetworkX graph object

    Returns:
        np.ndarray: NumPy adjacency matrix
    """
    try:
        return nx.to_numpy_array(G)
    except Exception as e:
        logger.error(f"Error converting to adjacency matrix: {e}")
        return None