"""
metis_parser.py

This module provides a function to load a graph in METIS file format and
construct an adjacency list representation. It supports both weighted and
unweighted graphs, converts the default 1‐based METIS indexing to 0‐based indexing,
and performs robust error handling.

The function returns a tuple:
    (adjacency_list, num_nodes, num_edges)

Example METIS header formats:
    Unweighted:      "4 6"
    Weighted:        "4 6 011"       -> (vertex weights absent, edge weights present)
    With vertex weights (ncon provided): "4 6 101 2"
    
Refer to Section 4.1.1 of the METIS Manual for details.
"""

import logging
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def load_graph_from_metis(file_path: str) -> Tuple[Dict[int, List[int]], int, int]:
    """
    Reads a graph in METIS format from the specified file, constructs an adjacency list,
    and returns a tuple (adjacency_list, num_nodes, num_edges).

    The function handles:
      - Skipping comment and empty lines.
      - Parsing the header for number of nodes, number of edges, and an optional format token.
      - Supporting weighted graphs:
           * If vertex weights are present, the first ncon tokens (ncon specified in header
             or defaulting to 1) in each vertex line are skipped.
           * If edge weights are present, the remaining tokens are expected in pairs
             (neighbor, weight) and only the neighbor (converted to 0-based index) is stored.
      - For unweighted graphs, each token (after skipping vertex weights if applicable)
        is interpreted as a neighbor.
      - Conversion from 1-based indexing to 0-based indexing.

    Parameters:
        file_path (str): Path to the METIS graph file.

    Returns:
        Tuple[Dict[int, List[int]], int, int]:
            - adjacency_list: A dictionary mapping each vertex (0-based) to a list of neighbor vertices.
              (Edge weights, if present, are ignored for connectivity purposes.)
            - num_nodes: Total number of nodes.
            - num_edges: Total number of undirected edges.

    Raises:
        ValueError: If the file is empty, the header is invalid, or if token parsing fails.
        IOError: If the file cannot be read.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Error reading file '{file_path}': {e}")
        raise

    # Filter out empty lines and comments (lines starting with '%')
    content_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('%'):
            continue
        content_lines.append(stripped)

    if not content_lines:
        logger.error(
            "METIS file is empty or contains only comments/whitespace.")
        raise ValueError("Empty METIS file.")

    # Parse header
    header_tokens = content_lines[0].split()
    if len(header_tokens) < 2:
        logger.error(
            "Invalid METIS header: requires at least two tokens (num_nodes and num_edges).")
        raise ValueError("Invalid METIS header: not enough tokens.")

    try:
        num_nodes = int(header_tokens[0])
        header_edge_count = int(header_tokens[1])
    except Exception as e:
        logger.error("Invalid numeric values in METIS header.")
        raise ValueError("Invalid header numbers.") from e

    # Determine if weights are provided
    vertex_weights = False
    edge_weights = False
    ncon = 1  # default: one vertex weight per vertex if weights are provided
    if len(header_tokens) >= 3:
        fmt = header_tokens[2]
        if len(fmt) >= 1:
            vertex_weights = (fmt[0] == '1')
        if len(fmt) >= 2:
            edge_weights = (fmt[1] == '1')
        # If vertex weights are present, ncon may be provided as the fourth token.
        if vertex_weights and len(header_tokens) >= 4:
            try:
                ncon = int(header_tokens[3])
            except Exception as e:
                logger.error("Invalid ncon value in header.")
                raise ValueError("Invalid ncon in header.") from e

    adjacency_list = {}

    if len(content_lines) - 1 < num_nodes:
        logger.error(
            "The number of vertex lines in the file is less than the expected number of nodes.")
        raise ValueError("Insufficient vertex lines in METIS file.")

    # Process each vertex line (vertices are expected in order; lines[1] is vertex 1, etc.)
    for i in range(num_nodes):
        line = content_lines[i + 1]
        tokens = line.split()
        token_index = 0

        # If vertex weights are provided, skip the first ncon tokens.
        if vertex_weights:
            if len(tokens) < ncon:
                logger.error(
                    f"Vertex {i + 1} does not contain enough tokens for vertex weights.")
                raise ValueError(
                    f"Insufficient vertex weight tokens for vertex {i + 1}.")
            token_index += ncon

        neighbors = []
        # Process remaining tokens
        if edge_weights:
            # Expect pairs: (neighbor, weight). Validate that there is an even number of tokens.
            remaining_tokens = tokens[token_index:]
            if len(remaining_tokens) % 2 != 0:
                logger.error(
                    f"Vertex {i + 1}: Expected an even number of tokens for edge weights, got {len(remaining_tokens)}.")
                raise ValueError(
                    f"Edge weights tokens count error in vertex {i + 1}.")
            for j in range(0, len(remaining_tokens), 2):
                try:
                    # Only neighbor is stored; subtract 1 for 0-based indexing.
                    neighbor = int(remaining_tokens[j]) - 1
                except Exception as e:
                    logger.error(
                        f"Vertex {i + 1}: Invalid neighbor token '{remaining_tokens[j]}'.")
                    raise ValueError(
                        f"Invalid neighbor token in vertex {i + 1}.") from e
                neighbors.append(neighbor)
        else:
            # Unweighted graph: each token represents a neighbor.
            for token in tokens[token_index:]:
                try:
                    neighbor = int(token) - 1
                except Exception as e:
                    logger.error(
                        f"Vertex {i + 1}: Invalid neighbor token '{token}'.")
                    raise ValueError(
                        f"Invalid neighbor token in vertex {i + 1}.") from e
                neighbors.append(neighbor)

        # Validate neighbor indices
        for neighbor in neighbors:
            if neighbor < 0 or neighbor >= num_nodes:
                logger.error(
                    f"Vertex {i + 1}: Neighbor index {neighbor + 1} is out of valid range (1 to {num_nodes}).")
                raise ValueError(
                    f"Neighbor index out of range in vertex {i + 1}.")

        adjacency_list[i] = neighbors

    # Determine the final number of edges.
    # Note: METIS header edge count (header_edge_count) may refer to undirected edges.
    # We compute the total number of neighbor entries.
    total_neighbor_entries = sum(len(neigh)
                                 for neigh in adjacency_list.values())
    if total_neighbor_entries == header_edge_count:
        final_num_edges = header_edge_count
    elif total_neighbor_entries == 2 * header_edge_count:
        final_num_edges = header_edge_count
    else:
        # Fall back to computing half the sum (assuming undirected double-listing)
        final_num_edges = total_neighbor_entries // 2
        logger.warning(
            f"Computed total neighbor entries ({total_neighbor_entries}) do not match the header edge count ({header_edge_count}). "
            f"Using computed edge count: {final_num_edges}."
        )

    logger.info(
        f"Loaded METIS graph from '{file_path}': {num_nodes} nodes, {final_num_edges} edges.")
    return adjacency_list, num_nodes, final_num_edges
