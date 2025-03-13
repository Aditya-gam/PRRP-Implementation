import metis
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def partition_graph_pymetis(adj_list: Dict[int, List[int]], num_partitions: int) -> Dict[int, Set[int]]:
    """
    Partitions a graph using PyMETIS into the specified number of partitions.

    Parameters:
        adj_list (Dict[int, List[int]]): The adjacency list of the graph.
        num_partitions (int): The number of partitions to divide the graph into.

    Returns:
        Dict[int, Set[int]]: A dictionary mapping partition indices to sets of node IDs.
    """
    if num_partitions <= 1:
        logger.error("Number of partitions must be greater than 1.")
        raise ValueError("Number of partitions must be greater than 1.")

    # Ensure nodes are sequential integers starting from 0 (required by METIS)
    node_map = {node: idx for idx, node in enumerate(sorted(adj_list.keys()))}
    reverse_map = {idx: node for node, idx in node_map.items()}

    # Convert adjacency list to METIS format (list of lists)
    metis_graph = [[node_map[neighbor] for neighbor in adj_list[node]]
                   for node in sorted(adj_list.keys())]

    try:
        logger.info(
            f"Partitioning graph using PyMETIS into {num_partitions} partitions...")
        _, partitions = metis.part_graph(metis_graph, nparts=num_partitions)
    except Exception as e:
        logger.error(f"PyMETIS partitioning failed: {e}")
        raise RuntimeError(f"PyMETIS failed due to: {e}")

    # Organize partitions into dictionary format
    partition_dict = {i: set() for i in range(num_partitions)}
    for idx, part_id in enumerate(partitions):
        original_node = reverse_map[idx]  # Map back to original node IDs
        partition_dict[part_id].add(original_node)

    logger.info("PyMETIS partitioning completed successfully.")
    return partition_dict
