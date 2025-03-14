"""
utils.py

This module contains utility functions for the P-Regionalization through Recursive Partitioning (PRRP)
algorithm. It provides functions for adjacency list construction (with parallelization when a GeoDataFrame
is used), graph analysis (e.g. connected components, articulation points), and includes a Disjoint Set
Union (DSU) implementation for efficient region merging.
"""

import logging
import random
import concurrent.futures
import heapq
from multiprocessing import Pool, cpu_count
import os
from typing import Dict, List, Set, Any, Tuple, Callable, Iterable
import geopandas as gpd
from shapely.geometry.base import BaseGeometry

# Global flag for parallel processing.
PARALLEL_PROCESSING_ENABLED = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class DisjointSetUnion:
    """
    Disjoint Set Union (Union-Find) implementation with path compression.
    """

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX


def _has_rook_adjacency(geom1: BaseGeometry, geom2: BaseGeometry) -> bool:
    """
    Checks if two geometries share a rook-adjacent boundary (i.e., a common edge).
    """
    if not geom1.touches(geom2):
        return False

    intersection = geom1.boundary.intersection(geom2.boundary)
    if intersection.is_empty:
        return False

    if intersection.geom_type in ['LineString', 'MultiLineString']:
        return True

    if intersection.geom_type == 'GeometryCollection':
        for geom in intersection.geoms:
            if geom.geom_type in ['LineString', 'MultiLineString']:
                return True

    return False


def construct_adjacency_list(areas: Any) -> Dict[Any, Set[Any]]:
    """
    Creates a graph adjacency list using rook adjacency for spatial data or by converting a
    pre-constructed graph-based input.

    For GeoDataFrame inputs, this function uses a ThreadPoolExecutor to parallelize the processing.
    **If the GeoDataFrame contains an 'id' column, that column is used as the index so that the keys
    in the returned adjacency list correspond to the actual area IDs rather than the default 0-based index.
    Note that the spatial index returns integer positions. These are used with .iloc to get the
    actual index label from the GeoDataFrame.**

    Parameters:
        areas (GeoDataFrame, list, or dict): Spatial areas with geometry information or a pre-built adjacency list.

    Returns:
        Dict[Any, Set[Any]]: Mapping from area identifiers to sets of adjacent area identifiers.

    Raises:
        TypeError: If the input type is unsupported.
        ValueError: If required geometry information is missing.
    """
    if isinstance(areas, dict):
        for key, value in areas.items():
            if not isinstance(value, set):
                areas[key] = set(value)
        logger.info(
            "Input is a dictionary; converted neighbor lists to sets in-place.")
        return areas

    if isinstance(areas, gpd.GeoDataFrame):
        # If the GeoDataFrame has an 'id' column, use it as the index.
        if 'id' in areas.columns:
            areas = areas.set_index('id')
        adj_list = {}

        def process_row(item):
            key, area = item  # key is the index label (e.g., 1,2,...)
            geom = area.geometry
            neighbors = set()
            # Use the spatial index for fast neighbor lookup.
            # Note: the intersection returns positional indices (0-based positions)
            possible_matches = list(areas.sindex.intersection(geom.bounds))
            for pos in possible_matches:
                # Retrieve the row by position using .iloc.
                other_row = areas.iloc[pos]
                other_label = other_row.name
                if other_label == key:
                    continue
                other_geom = other_row.geometry
                if _has_rook_adjacency(geom, other_geom):
                    neighbors.add(other_label)
            return key, neighbors

        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            results = executor.map(process_row, areas.iterrows())
        for key, nbrs in results:
            adj_list[key] = nbrs
        logger.info("Adjacency list constructed in parallel from GeoDataFrame.")
        return adj_list

    if isinstance(areas, list):
        # Processing for list of dicts (sequentially).
        adj_list = {}
        if not all(isinstance(area, dict) for area in areas):
            logger.error(
                "Unsupported list element type; expected dicts with geometry information.")
            raise TypeError(
                "Unsupported type for areas. Expected GeoDataFrame, list of dicts, or dict.")
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
                "Adjacency list constructed from list of dicts based on geometries.")
            return adj_list

    logger.error(
        "Unsupported type for areas. Expected GeoDataFrame, list of dicts, or dict.")
    raise TypeError("Unsupported type for areas.")


def parallel_execute(function: Callable[[Any], Any],
                     data: Iterable[Any],
                     num_threads: int = 1,
                     use_multiprocessing: bool = False) -> List[Any]:
    """
    Executes a function in parallel over the given data.
    Automatically selects between multi-threading and multiprocessing based on the flag.

    Parameters:
        function (Callable[[Any], Any]): The function to apply.
        data (Iterable[Any]): Data items to process.
        num_threads (int): Number of threads/processes to use.
        use_multiprocessing (bool): Whether to use multiprocessing.

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
        if not results:
            logger.warning("No results returned from the function execution.")
        else:
            logger.debug(f"Results: {results}")

        return results


def find_articulation_points(G: Dict[int, List[int]]) -> Set[int]:
    """
    Computes the articulation points of a graph using Tarjan’s Algorithm in O(V + E) time.

    Parameters:
        G (Dict[int, List[int]]): Graph represented as an adjacency list.

    Returns:
        Set[int]: The set of articulation points.
    """
    adj = {u: set(neighbors) for u, neighbors in G.items()}
    if not adj:
        return set()

    disc: Dict[int, int] = {}
    low: Dict[int, int] = {}
    parent: Dict[int, int] = {}
    ap: Set[int] = set()
    time_counter = 0
    child_count: Dict[int, int] = {}
    stack_frame = []

    for u in adj:
        if u in disc:
            continue
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
                    if parent[current] == par and parent[par] is not None and low[current] >= disc[par]:
                        ap.add(par)
                else:
                    if child_count[current] > 1:
                        ap.add(current)

    return ap


def find_connected_components(adj_list: Dict[Any, List[Any]]) -> List[Set[Any]]:
    """
    Finds all connected components in a graph using DFS.

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.

    Returns:
        List[Set[Any]]: A list of connected components (each component is a set of nodes).
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
    Determines whether a node is an articulation point using Tarjan's Algorithm.

    A node is an articulation point if its removal increases the number of connected components.

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.
        node (Any): The node to check.

    Returns:
        bool: True if the node is an articulation point; False otherwise.
    """
    if node not in adj_list:
        logger.error(f"Node {node} not found in the adjacency list.")
        raise KeyError(f"Node {node} not found in the adjacency list.")

    if len(adj_list[node]) <= 1:
        logger.info(
            f"Node {node} is a leaf and cannot be an articulation point.")
        return False

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
    If removal disconnects the graph, reassigns the node to the largest connected component.

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.
        node (Any): The articulation point to remove.

    Returns:
        Dict[Any, List[Any]]: The updated adjacency list.
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
                new_adj[neighbor].append(node)
        logger.warning(
            f"Articulation node {node} removed and reassigned to maintain connectivity.")
    else:
        new_adj[node] = [original_neighbors[0]]
        new_adj[original_neighbors[0]].append(node)
        logger.info(
            f"Node {node} was removed but reassigned to maintain connectivity.")

    return new_adj


def random_seed_selection(adj_list: Dict[Any, List[Any]], assigned_regions: Set[Any], method: str = "gapless") -> Any:
    """
    Selects a seed node efficiently from unassigned nodes using a heap-based approach.
    This function computes connectivity scores for unassigned nodes once and uses a min-heap to retrieve
    the top candidates quickly. The score is defined as the number of neighbors in assigned_regions.

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.
        assigned_regions (Set[Any]): Nodes that are already assigned.
        method (str): Selection method. Currently, only "gapless" is supported.

    Returns:
        Any: A selected seed node.

    Raises:
        ValueError: If no unassigned nodes are available or if the method is unknown.
    """
    unassigned = set(adj_list.keys()) - assigned_regions
    if not unassigned:
        logger.error("No unassigned nodes available for seed selection.")
        raise ValueError("No unassigned nodes available.")

    if method == "gapless":
        # Build a heap of (negative_score, node) for the top candidates.
        # The score is the number of neighbors in assigned_regions.
        heap = []
        for node in unassigned:
            score = sum(1 for nbr in adj_list[node] if nbr in assigned_regions)
            # We want high scores to come first, so use negative score.
            heapq.heappush(heap, (-score, node))
        # Retrieve the top candidate from the heap (if there are several with equal score, one is chosen randomly).
        # For extra randomness, we extract up to 10 best candidates and pick one.
        top_candidates = []
        for _ in range(min(10, len(heap))):
            top_candidates.append(heapq.heappop(heap)[1])
        chosen = random.choice(top_candidates)
        logger.info(
            f"Selected gapless seed: {chosen} from top candidates {top_candidates}")

        return chosen
    else:
        logger.error(f"Unknown seed selection method: {method}")
        raise ValueError(f"Unknown seed selection method: {method}")


def load_graph_from_metis(file_path: str) -> Dict[int, List[int]]:
    """
    Reads a graph in METIS format and converts it into an adjacency list.

    Parameters:
        file_path (str): Path to the METIS format file.

    Returns:
        Dict[int, List[int]]: Adjacency list with 1-based node indices.

    Raises:
        ValueError: If the file is empty, header is invalid, or malformed.
    """
    adj_list: Dict[int, List[int]] = {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            logger.error("METIS file is empty.")
            raise ValueError("Empty METIS file.")

        content_lines = [line.strip() for line in lines if line.strip()
                         and not line.strip().startswith('%')]
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

        if len(content_lines) - 1 < num_nodes:
            logger.error("Insufficient vertex lines in METIS file.")
            raise ValueError("Insufficient vertex lines in METIS file.")

        for i in range(num_nodes):
            line = content_lines[i + 1]
            tokens = line.split()
            neighbors = []
            for token in tokens:
                try:
                    neighbor = int(token)
                except Exception as e:
                    logger.error(
                        f"Vertex {i+1}: Invalid neighbor token '{token}'.")
                    raise ValueError(
                        f"Invalid neighbor token in vertex {i+1}.") from e
                if neighbor != i + 1:
                    neighbors.append(neighbor)
            for neighbor in neighbors:
                if neighbor < 1 or neighbor > num_nodes:
                    logger.error(
                        f"Vertex {i+1}: Neighbor index {neighbor} out of range (1 to {num_nodes}).")
                    raise ValueError(
                        f"Neighbor index out of range in vertex {i+1}.")
            adj_list[i + 1] = neighbors

        total_neighbor_entries = sum(len(nlist) for nlist in adj_list.values())
        if total_neighbor_entries == header_edge_count:
            final_num_edges = header_edge_count
        elif total_neighbor_entries == 2 * header_edge_count:
            final_num_edges = header_edge_count
        else:
            final_num_edges = total_neighbor_entries // 2
            logger.warning(
                f"Computed total neighbor entries ({total_neighbor_entries}) do not match header edge count ({header_edge_count}). Using computed edge count: {final_num_edges}.")

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
        file_path (str): File path for the METIS graph.
        adj_list (Dict[int, List[int]]): The adjacency list.

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
    Identifies boundary areas of a region.

    Parameters:
        region (Set[Any]): Nodes in the region.
        adj_list (Dict[Any, List[Any]]): Graph's adjacency list.

    Returns:
        Set[Any]: Boundary nodes with at least one neighbor outside the region.
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
    Computes discovery and low-link values for nodes in the graph using DFS.

    Parameters:
        adj_list (Dict[Any, List[Any]]): Graph's adjacency list.

    Returns:
        Tuple[Dict[Any, int], Dict[Any, int]]: (disc, low) mappings.
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
    Computes the degree list for each node and establishes parent-child relationships.

    Parameters:
        G (Dict[int, List[int]]): Graph adjacency list.

    Returns:
        Dict[int, List[int]]: Mapping of each node to its child nodes.
    """
    degrees = {node: len(neighbors) for node, neighbors in G.items()}
    is_parent = {}
    for node, neighbors in G.items():
        if not neighbors:
            is_parent[node] = False
            continue
        neighbor_degrees = [degrees[nb] for nb in neighbors if nb in degrees]
        sorted_degs = sorted(neighbor_degrees)
        n = len(sorted_degs)
        median = sorted_degs[n // 2] if n % 2 == 1 else (
            sorted_degs[n // 2 - 1] + sorted_degs[n // 2]) / 2
        is_parent[node] = degrees[node] > median

    degree_list = {node: [] for node in G}
    processed_edges = set()
    for u in G:
        for v in G[u]:
            if v not in G:
                continue
            if u not in G[v]:
                continue
            edge = tuple(sorted((u, v)))
            if edge in processed_edges:
                continue
            processed_edges.add(edge)
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
    Executes a function in parallel over the given data if enabled.

    Parameters:
        function (Callable[[Any], Any]): The function to apply.
        data (Iterable[Any]): Data items to process.
        num_threads (int): Number of threads/processes to use.
        use_multiprocessing (bool): Whether to use multiprocessing.

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
