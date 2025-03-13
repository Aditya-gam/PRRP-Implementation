"""
utils.py

This module contains utility functions for the P-Regionalization through Recursive Partitioning (PRRP)
algorithm as described in "Statistical Inference for Spatial Regionalization" (SIGSPATIAL 2023)
by Hussah Alrashid, Amr Magdy, and Sergio Rey.

Functions:
    - construct_adjacency_list(areas)
    - find_connected_components(adj_list)
    - is_articulation_point(adj_list, node)
    - find_articulation_points(G)
    - remove_articulation_area(adj_list, node)
    - random_seed_selection(adj_list, assigned_regions, method="gapless")
    - load_graph_from_metis(file_path)
    - save_graph_to_metis(file_path, adj_list)
    - find_boundary_areas(region, adj_list)
    - calculate_low_link_values(adj_list)
    - parallel_execute(function, data, num_threads=1, use_multiprocessing=False)
"""

import logging
import random
import concurrent.futures
from multiprocessing import Pool, cpu_count
import os
import copy
from typing import Dict, List, Set, Any, Tuple, Callable, Iterable
import geopandas as gpd
from shapely.geometry.base import BaseGeometry

# Global flag to enable parallel processing. Set to True to allow parallel execution.
PARALLEL_PROCESSING_ENABLED = False

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust as needed.
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


def _has_rook_adjacency(geom1: BaseGeometry, geom2: BaseGeometry) -> bool:
    """
    Check if two geometries share a rook-adjacent boundary (i.e., share a common edge).

    Parameters:
        geom1 (BaseGeometry): The first geometry.
        geom2 (BaseGeometry): The second geometry.

    Returns:
        bool: True if the geometries are rook adjacent, else False.
    """
    if not geom1.touches(geom2):
        return False

    intersection = geom1.boundary.intersection(geom2.boundary)
    if intersection.is_empty:
        return False

    # Check if the intersection is (or contains) a LineString (edge) rather than a point.
    if intersection.geom_type in ['LineString', 'MultiLineString']:
        return True

    if intersection.geom_type == 'GeometryCollection':
        for geom in intersection.geoms:
            if geom.geom_type in ['LineString', 'MultiLineString']:
                return True

    return False


def construct_adjacency_list(areas: Any) -> Dict[Any, Set[Any]]:
    """
    Creates a graph adjacency list using rook adjacency (for spatial data) or by directly
    converting a pre-constructed graph-based input.

    This function supports three forms of input:
      1. A pre-constructed adjacency list (dictionary): Converts each neighbor list to a set.
      2. A GeoDataFrame (spatial PRRP): Uses spatial indexing and geometry intersections.
      3. A list of dictionaries (custom spatial data): Constructs either a complete graph
         (if all geometries are None) or computes adjacency based on geometries.

    Parameters:
        areas (GeoDataFrame, list, or dict): Spatial areas with geometry information or a pre-built adjacency list.

    Returns:
        Dict[Any, Set[Any]]: A dictionary mapping each area identifier to a set of adjacent area identifiers.

    Raises:
        TypeError: If the input is not a GeoDataFrame, list, or dictionary.
        ValueError: If a required geometry is missing.
    """
    # Case 1: Input is already a dictionary (graph-based PRRP)
    if isinstance(areas, dict):
        new_adj_list = {}
        for key, value in areas.items():
            # Convert neighbor lists to sets for efficient lookups.
            new_adj_list[key] = value if isinstance(value, set) else set(value)
        logger.info(
            "Input is a dictionary; returning pre-constructed adjacency list.")
        return new_adj_list

    adj_list: Dict[Any, Set[Any]] = {}

    # Case 2: Input is a GeoDataFrame (spatial PRRP)
    if isinstance(areas, gpd.GeoDataFrame):
        try:
            sindex = areas.sindex
        except Exception as e:
            logger.error(f"Spatial index creation failed: {e}")
            raise

        for idx, area in areas.iterrows():
            geom = area.geometry
            adj_list[idx] = set()
            possible_matches = list(sindex.intersection(geom.bounds))
            for other_idx in possible_matches:
                if other_idx == idx:
                    continue
                other_geom = areas.loc[other_idx].geometry
                if _has_rook_adjacency(geom, other_geom):
                    adj_list[idx].add(other_idx)

    # Case 3: Input is a list of dictionaries (custom spatial data)
    elif isinstance(areas, list):
        # NEW: Check that each element in the list is a dictionary.
        if not all(isinstance(area, dict) for area in areas):
            logger.error(
                "Unsupported list element type in areas. Expected each element to be a dict with geometry information.")
            raise TypeError(
                "Unsupported type for areas. Expected GeoDataFrame, list of dicts, or dict.")

        # If all areas lack geometry, construct a complete graph.
        if all(area.get('geometry') is None for area in areas):
            for area in areas:
                area_id = area.get('id')
                adj_list[area_id] = {
                    other.get('id') for other in areas if other.get('id') != area_id}
            logger.info(
                "Adjacency list constructed as a complete graph (dummy geometry).")
            return adj_list
        else:
            n = len(areas)
            for i in range(n):
                area_i = areas[i]
                id_i = area_i.get('id', i)
                geom_i = area_i.get('geometry')
                if geom_i is None:
                    logger.error(f"Area with id {id_i} has no geometry.")
                    raise ValueError(f"Area with id {id_i} has no geometry.")
                adj_list[id_i] = set()
                for j in range(n):
                    if i == j:
                        continue
                    area_j = areas[j]
                    id_j = area_j.get('id', j)
                    geom_j = area_j.get('geometry')
                    if geom_j is None:
                        logger.error(f"Area with id {id_j} has no geometry.")
                        raise ValueError(
                            f"Area with id {id_j} has no geometry.")
                    if _has_rook_adjacency(geom_i, geom_j):
                        adj_list[id_i].add(id_j)
            logger.info(
                "Adjacency list constructed successfully based on geometries.")
    else:
        logger.error(
            "Unsupported type for areas. Expected GeoDataFrame, list, or dict.")
        raise TypeError(
            "Unsupported type for areas. Expected GeoDataFrame, list of dicts, or dict.")

    return adj_list


def find_articulation_points(G: Dict[int, List[int]]) -> Set[int]:
    """
    Computes articulation points in a graph using Tarjan’s Algorithm implemented via an
    iterative DFS approach. This version processes every connected component separately.

    Parameters:
        G (Dict[int, List[int]]): Graph represented as an adjacency list where keys are node IDs
                                  and values are lists of adjacent node IDs.

    Returns:
        Set[int]: Set of nodes that are articulation points.
    """
    # Convert neighbor lists to sets (if they aren’t already) for O(1) lookups.
    adj = {u: set(neighbors) for u, neighbors in G.items()}
    if not adj:
        return set()

    disc: Dict[int, int] = {}
    low: Dict[int, int] = {}
    parent: Dict[int, int] = {}
    ap: Set[int] = set()
    time_counter = 0
    child_count: Dict[int, int] = {}

    # Stack for iterative DFS: each element is a tuple (node, neighbor_iterator)
    stack_frame = []

    # Process every node to cover each connected component.
    for u in adj:
        if u in disc:
            continue  # Already visited in a previous DFS.
        parent[u] = None
        disc[u] = time_counter
        low[u] = time_counter
        time_counter += 1
        child_count[u] = 0
        stack_frame.append((u, iter(adj[u])))

        while stack_frame:
            current, neighbors_iter = stack_frame[-1]
            try:
                v = next(neighbors_iter)
                if v not in disc:
                    parent[v] = current
                    child_count[current] = child_count.get(current, 0) + 1
                    disc[v] = time_counter
                    low[v] = time_counter
                    time_counter += 1
                    child_count[v] = 0
                    stack_frame.append((v, iter(adj[v])))
                elif v != parent[current]:
                    low[current] = min(low[current], disc[v])
            except StopIteration:
                stack_frame.pop()
                if stack_frame:
                    par, _ = stack_frame[-1]
                    low[par] = min(low[par], low[current])
                    # Only mark par as an articulation point if par is not the root.
                    if parent[current] == par and parent[par] is not None and low[current] >= disc[par]:
                        ap.add(par)
                else:
                    # 'current' is the root of the connected component.
                    if child_count[current] > 1:
                        ap.add(current)
    return ap


def find_connected_components(adj_list: Dict[Any, List[Any]]) -> List[Set[Any]]:
    """
    Finds all connected components in a graph using DFS.

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.

    Returns:
        List[Set[Any]]: A list of sets, where each set contains nodes in one connected component.
    """
    visited: Set[Any] = set()
    components: List[Set[Any]] = []

    for node in adj_list.keys():
        if node not in visited:
            component: Set[Any] = set()
            stack: List[Any] = [node]
            while stack:
                n = stack.pop()
                if n not in visited:
                    visited.add(n)
                    component.add(n)
                    for neighbor in adj_list.get(n, []):
                        if neighbor not in visited:
                            stack.append(neighbor)
            components.append(component)

    logger.info(f"Found {len(components)} connected component(s).")
    return components


def is_articulation_point(adj_list: Dict[Any, List[Any]], node: Any) -> bool:
    """
    Determines if a given node is an articulation point using Tarjan's Algorithm.

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.
        node (Any): The node to check.

    Returns:
        bool: True if the node is an articulation point; False otherwise.
    """
    if node not in adj_list:
        logger.error(f"Node {node} not found in the adjacency list.")
        raise KeyError(f"Node {node} not found in the adjacency list.")

    def tarjan_ap_util(v: Any, parent: Any, disc: Dict[Any, int],
                       low: Dict[Any, int], time: List[int], ap: Set[Any]) -> None:
        children = 0
        disc[v] = low[v] = time[0]
        time[0] += 1

        for w in adj_list[v]:
            if w not in disc:
                children += 1
                tarjan_ap_util(w, v, disc, low, time, ap)
                low[v] = min(low[v], low[w])
                if parent is None and children > 1:
                    ap.add(v)
                if parent is not None and low[w] >= disc[v]:
                    ap.add(v)
            elif w != parent:
                low[v] = min(low[v], disc[w])

    disc: Dict[Any, int] = {}
    low: Dict[Any, int] = {}
    time: List[int] = [0]
    ap: Set[Any] = set()

    for v in adj_list.keys():
        if v not in disc:
            tarjan_ap_util(v, None, disc, low, time, ap)

    is_ap = node in ap
    logger.info(
        f"Node {node} is {'an' if is_ap else 'not an'} articulation point.")

    return is_ap


def remove_articulation_area(adj_list: Dict[Any, List[Any]], node: Any) -> Dict[Any, List[Any]]:
    """
    Removes an articulation point from the graph and reassigns it to maintain connectivity.
    If removal disconnects the graph, the node is reassigned to the largest connected component;
    otherwise, it is reassigned to one of its original neighbors.

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.
        node (Any): The articulation point to remove.

    Returns:
        Dict[Any, List[Any]]: The updated adjacency list after removal and reassignment.
    """
    if node not in adj_list:
        logger.error(f"Node {node} not found in the adjacency list.")
        raise KeyError(f"Node {node} not in adjacency list.")

    new_adj = {k: list(v) for k, v in adj_list.items()}
    original_neighbors = new_adj[node]

    for neighbor in original_neighbors:
        if node in new_adj[neighbor]:
            new_adj[neighbor].remove(node)
    del new_adj[node]

    components = find_connected_components(new_adj)
    if len(components) > 1:
        largest_component = max(components, key=len)
        new_adj[node] = []
        for neighbor in original_neighbors:
            if neighbor in largest_component:
                new_adj[node].append(neighbor)
                # Ensure bidirectional connection
                new_adj[neighbor].append(node)
        logger.warning(
            f"Articulation node {node} removed and reassigned to maintain connectivity.")
    else:
        new_adj[node] = [original_neighbors[0]]
        new_adj[original_neighbors[0]].append(node)
        logger.info(
            f"Node {node} was removed but reassigned to maintain connectivity.")

    return new_adj


def random_seed_selection(adj_list: Dict[Any, List[Any]], assigned_regions: Set[Any],
                          method: str = "gapless") -> Any:
    """
    Selects a seed node from the unassigned nodes using the "gapless" method.
    A candidate seed is one that has at least one neighbor already assigned; if none exists,
    a random unassigned node is returned.

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.
        assigned_regions (Set[Any]): Nodes already assigned to regions.
        method (str, optional): Seed selection method (default "gapless").

    Returns:
        Any: The selected seed node.

    Raises:
        ValueError: If no unassigned nodes are available or the method is unknown.
    """
    unassigned = set(adj_list.keys()) - assigned_regions
    if not unassigned:
        logger.error("No unassigned nodes available for seed selection.")
        raise ValueError("No unassigned nodes available.")

    if method == "gapless":
        candidate_seeds = {node for node in unassigned
                           if any(neighbor in assigned_regions for neighbor in adj_list[node])}
        if candidate_seeds:
            chosen = random.choice(list(candidate_seeds))
            logger.info(f"Selected gapless seed: {chosen}")
            return chosen
        chosen = random.choice(list(unassigned))
        logger.info(
            f"No gapless candidate found, selected random seed: {chosen}")
        return chosen
    else:
        logger.error(f"Unknown seed selection method: {method}")
        raise ValueError(f"Unknown seed selection method: {method}")


def load_graph_from_metis(file_path: str) -> Dict[int, List[int]]:
    """
    Reads a graph in METIS format and converts it into an adjacency list.

    This function expects that the METIS file uses 1-based indexing.
    The output will be a dictionary with keys 1..n and neighbor lists
    that are also 1-indexed.

    Parameters:
        file_path (str): Path to the METIS format file.

    Returns:
        Dict[int, List[int]]: The adjacency list representation of the graph,
                              with 1-based node indices.

    Raises:
        ValueError: If the file is empty, the header is invalid, or the file is malformed.
        Exception: For other I/O or parsing errors.
    """
    adj_list: Dict[int, List[int]] = {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            logger.error("METIS file is empty.")
            raise ValueError("Empty METIS file.")

        # Remove blank lines and comment lines (starting with '%')
        content_lines = [
            line.strip() for line in lines if line.strip() and not line.strip().startswith('%')
        ]
        if not content_lines:
            logger.error(
                "METIS file is empty or contains only comments/whitespace.")
            raise ValueError("Empty METIS file.")

        header_tokens = content_lines[0].split()
        if len(header_tokens) < 2:
            logger.error(
                "Invalid METIS header: requires at least two tokens (num_nodes and num_edges).")
            raise ValueError("Invalid METIS header: not enough tokens.")
        num_nodes = int(header_tokens[0])
        header_edge_count = int(header_tokens[1])

        # Ensure there are enough lines for all nodes.
        if len(content_lines) - 1 < num_nodes:
            logger.error("Insufficient vertex lines in METIS file.")
            raise ValueError("Insufficient vertex lines in METIS file.")

        # Process each node (using 1-based indexing)
        for i in range(num_nodes):
            # Node id will be i+1.
            line = content_lines[i + 1]
            tokens = line.split()
            neighbors = []
            for token in tokens:
                try:
                    neighbor = int(token)  # DO NOT subtract 1.
                except Exception as e:
                    logger.error(
                        f"Vertex {i+1}: Invalid neighbor token '{token}'.")
                    raise ValueError(
                        f"Invalid neighbor token in vertex {i+1}.") from e
                # Avoid self-loops (if neighbor equals the current node id)
                if neighbor != i + 1:
                    neighbors.append(neighbor)
            # Validate that neighbor indices are in the range 1..num_nodes.
            for neighbor in neighbors:
                if neighbor < 1 or neighbor > num_nodes:
                    logger.error(
                        f"Vertex {i+1}: Neighbor index {neighbor} out of range (1 to {num_nodes}).")
                    raise ValueError(
                        f"Neighbor index out of range in vertex {i+1}.")
            adj_list[i + 1] = neighbors

        total_neighbor_entries = sum(len(nlist) for nlist in adj_list.values())
        # The header may list the number of edges (which might be half the sum if undirected)
        if total_neighbor_entries == header_edge_count:
            final_num_edges = header_edge_count
        elif total_neighbor_entries == 2 * header_edge_count:
            final_num_edges = header_edge_count
        else:
            final_num_edges = total_neighbor_entries // 2
            logger.warning(
                f"Computed total neighbor entries ({total_neighbor_entries}) do not match header edge count ({header_edge_count}). Using computed edge count: {final_num_edges}."
            )

        logger.info(
            f"Loaded METIS graph from '{file_path}': {num_nodes} nodes, {final_num_edges} edges.")
        return adj_list

    except Exception as e:
        logger.error(f"Failed to load METIS graph from {file_path}: {e}")
        raise


def save_graph_to_metis(file_path: str, adj_list: Dict[int, List[int]]) -> None:
    """
    Saves an adjacency list in METIS format.

    Parameters:
        file_path (str): The file path to save the METIS graph.
        adj_list (Dict[int, List[int]]): The adjacency list of the graph.

    Returns:
        None
    """
    try:
        num_nodes = len(adj_list)
        num_edges = sum(len(neighbors) for neighbors in adj_list.values()) // 2
        with open(file_path, 'w') as f:
            f.write(f"{num_nodes} {num_edges}\n")
            for i in range(1, num_nodes + 1):
                neighbors = adj_list.get(i, [])
                f.write(" ".join(map(str, neighbors)) + "\n")
        logger.info(
            f"Graph saved to {file_path} in METIS format successfully.")
    except Exception as e:
        logger.error(
            f"Failed to save graph to METIS format at {file_path}: {e}")
        raise


def find_boundary_areas(region: Set[Any], adj_list: Dict[Any, List[Any]]) -> Set[Any]:
    """
    Identifies boundary areas of a region. A boundary area is a node with at least one neighbor outside the region.

    Parameters:
        region (Set[Any]): The set of nodes in the region.
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.

    Returns:
        Set[Any]: The set of boundary nodes.
    """
    boundary: Set[Any] = set()
    for node in region:
        for neighbor in adj_list.get(node, []):
            if neighbor not in region:
                boundary.add(node)
                break
    logger.info(f"Identified {len(boundary)} boundary area(s) in the region.")
    return boundary


def calculate_low_link_values(adj_list: Dict[Any, List[Any]]) -> Tuple[Dict[Any, int], Dict[Any, int]]:
    """
    Computes discovery and low-link values for all nodes in the graph using DFS.
    These values are used for detecting articulation points via Tarjan's algorithm.

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.

    Returns:
        Tuple[Dict[Any, int], Dict[Any, int]]:
            - disc: Mapping of node to discovery time.
            - low: Mapping of node to its low-link value.
    """
    disc: Dict[Any, int] = {}
    low: Dict[Any, int] = {}
    time: List[int] = [0]

    def dfs(u: Any, parent: Any) -> None:
        disc[u] = low[u] = time[0]
        time[0] += 1
        for v in adj_list[u]:
            if v not in disc:
                dfs(v, u)
                low[u] = min(low[u], low[v])
            elif v != parent:
                low[u] = min(low[u], disc[v])

    for node in adj_list.keys():
        if node not in disc:
            dfs(node, None)

    logger.info("Calculated low-link values for all nodes.")
    return disc, low


def compute_degree_list(G: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """
    Computes the degree list for each node and identifies parent-child relationships.

    Parameters:
        G (Dict[int, List[int]]): Graph adjacency list representation.

    Returns:
        Dict[int, List[int]]: Dictionary mapping each node to its list of nested (child) nodes.

    Explanation:
    - Computes the degree (number of neighbors) for each node.
    - Identifies parent nodes as nodes with a degree higher than the median degree of their neighbors.
    - A node is assigned as a child if it is adjacent to a parent and has a lower degree.
    - Only symmetric edges (i.e., edges where u is in G[v] and v is in G[u]) are processed.
    - Isolated nodes (degree = 0) or nodes without a clear degree difference remain unassigned.
    - Optimized for large-scale graphs using O(V + E) dictionary lookups.
    """
    # Step 1: Compute the degree of each node
    degrees = {node: len(neighbors) for node, neighbors in G.items()}

    # Step 2: Identify potential parent nodes.
    # A node is considered a parent if its degree is greater than the median degree of its neighbors.
    is_parent = {}
    for node, neighbors in G.items():
        if not neighbors:  # Isolated node: cannot be a parent
            is_parent[node] = False
            continue

        # Gather the degrees of all neighboring nodes
        neighbor_degrees = [degrees[nb] for nb in neighbors if nb in degrees]
        sorted_degs = sorted(neighbor_degrees)
        n = len(sorted_degs)
        # Compute the median of neighbor degrees
        if n % 2 == 1:
            median = sorted_degs[n // 2]
        else:
            median = (sorted_degs[n // 2 - 1] + sorted_degs[n // 2]) / 2

        # Mark as parent if the node's degree is strictly greater than the median
        is_parent[node] = degrees[node] > median

    # Step 3: Initialize the degree list dictionary with an empty list for each node.
    degree_list = {node: [] for node in G}

    # Step 4: Process each undirected (symmetric) edge only once.
    processed_edges = set()
    for u in G:
        for v in G[u]:
            # Ensure both endpoints are valid nodes in the graph.
            if v not in G:
                continue
            # Enforce symmetry: process the edge only if u is in G[v]
            if u not in G[v]:
                continue

            # Create a unique representation for the undirected edge.
            edge = tuple(sorted((u, v)))
            if edge in processed_edges:
                continue
            processed_edges.add(edge)

            # Establish parent-child relationship:
            # If one node is a parent and the other is not—and the parent's degree is higher—assign the non-parent as a child.
            if is_parent.get(u, False) and (not is_parent.get(v, False)) and (degrees[u] > degrees[v]):
                degree_list[u].append(v)
            elif is_parent.get(v, False) and (not is_parent.get(u, False)) and (degrees[v] > degrees[u]):
                degree_list[v].append(u)

    return degree_list


def parallel_execute(function: Callable[[Any], Any],
                     data: Iterable[Any],
                     num_threads: int = 1,
                     use_multiprocessing: bool = False) -> List[Any]:
    """
    Executes a function in parallel over the provided data if parallel processing is enabled
    and num_threads > 1. Uses multiprocessing if specified; otherwise, uses threading.

    Parameters:
        function (Callable[[Any], Any]): The function to apply.
        data (Iterable[Any]): The data items to process.
        num_threads (int, optional): Number of threads/processes to use (default 1).
        use_multiprocessing (bool, optional): Whether to use multiprocessing (default False).

    Returns:
        List[Any]: List of results.
    """
    if num_threads > 1 and PARALLEL_PROCESSING_ENABLED:
        if use_multiprocessing:
            logger.info("Using multiprocessing for parallel execution.")
            with Pool(min(num_threads, cpu_count())) as pool:
                results = pool.map(function, list(data))
        else:
            logger.info("Using threading for parallel execution.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(executor.map(function, data))
        logger.info("Parallel execution completed successfully.")
        return results
    else:
        logger.info(
            "Parallel processing disabled or num_threads <= 1. Executing sequentially.")
        results = [function(item) for item in data]
        logger.info("Sequential execution completed successfully.")
        return results
