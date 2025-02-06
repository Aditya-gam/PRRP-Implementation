"""
utils.py

This module contains utility functions for the P-Regionalization through Recursive Partitioning (PRRP)
algorithm as described in "Statistical Inference for Spatial Regionalization" (SIGSPATIAL 2023)
by Hussah Alrashid, Amr Magdy, and Sergio Rey.

Functions:
    - construct_adjacency_list(areas)
    - find_connected_components(adj_list)
    - is_articulation_point(adj_list, node)
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


def construct_adjacency_list(areas: Any) -> Dict[Any, List[Any]]:
    """
    Creates a graph adjacency list using rook adjacency (i.e., regions that share a boundary).

    This function supports input as a GeoDataFrame (from geopandas) or a list of objects that have
    at least 'geometry' and an identifier (accessible via index or attribute).

    Parameters:
        areas (GeoDataFrame or list): Spatial areas with geometry information.

    Returns:
        Dict[Any, List[Any]]: A dictionary mapping each area identifier to a list of
                              spatially contiguous (adjacent) area identifiers.
    """
    adj_list: Dict[Any, List[Any]] = {}

    if isinstance(areas, gpd.GeoDataFrame):
        try:
            sindex = areas.sindex
        except Exception as e:
            logger.error(f"Spatial index creation failed: {e}")
            raise

        for idx, area in areas.iterrows():
            geom = area.geometry
            adj_list[idx] = []
            possible_matches_index = list(sindex.intersection(geom.bounds))
            for other_idx in possible_matches_index:
                if other_idx == idx:
                    continue
                other_geom = areas.loc[other_idx].geometry
                if _has_rook_adjacency(geom, other_geom):
                    adj_list[idx].append(other_idx)
    elif isinstance(areas, list):
        n = len(areas)
        for i in range(n):
            area_i = areas[i]
            id_i = area_i.get('id', i)
            geom_i = area_i.get('geometry')
            if geom_i is None:
                logger.error(f"Area with id {id_i} has no geometry.")
                raise ValueError(f"Area with id {id_i} has no geometry.")
            adj_list[id_i] = []
            for j in range(n):
                if i == j:
                    continue
                area_j = areas[j]
                id_j = area_j.get('id', j)
                geom_j = area_j.get('geometry')
                if geom_j is None:
                    logger.error(f"Area with id {id_j} has no geometry.")
                    raise ValueError(f"Area with id {id_j} has no geometry.")
                if _has_rook_adjacency(geom_i, geom_j):
                    adj_list[id_i].append(id_j)
    else:
        logger.error("Unsupported type for areas. Expected GeoDataFrame or list.")
        raise TypeError("Unsupported type for areas. Expected GeoDataFrame or list.")

    logger.info("Adjacency list constructed successfully.")
    return adj_list


def find_connected_components(adj_list: Dict[Any, List[Any]]) -> List[Set[Any]]:
    """
    Finds all spatially connected components in a graph using depth-first search (DFS).

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.

    Returns:
        List[Set[Any]]: A list of sets, where each set contains spatially connected nodes.
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
    Determines if a given node is an articulation point in the graph using Tarjan's Algorithm.

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.
        node (Any): The node to check.

    Returns:
        bool: True if the node is an articulation point, else False.
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
    logger.info(f"Node {node} is {'an' if is_ap else 'not an'} articulation point.")
    return is_ap


def remove_articulation_area(adj_list: Dict[Any, List[Any]], node: Any) -> Dict[Any, List[Any]]:
    """
    Removes an articulation point from the graph and reassigns it to maintain spatial connectivity.
    If removal of the node disconnects the graph, the node is reassigned to the largest connected
    component by reconnecting it to neighbors within that component.

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.
        node (Any): The articulation point to remove and reassign.

    Returns:
        Dict[Any, List[Any]]: The updated adjacency list after removal and reassignment.
    """
    if node not in adj_list:
        logger.error(f"Node {node} not found in the adjacency list.")
        raise KeyError(f"Node {node} not in adjacency list.")

    new_adj = {k: list(v) for k, v in adj_list.items()}

    for neighbor in new_adj[node]:
        if node in new_adj[neighbor]:
            new_adj[neighbor].remove(node)
    del new_adj[node]

    components = find_connected_components(new_adj)
    if len(components) <= 1:
        logger.info("Removal of the node did not disconnect the graph.")
        return new_adj

    largest_component = max(components, key=len)
    new_adj[node] = []
    original_neighbors = adj_list[node]
    for neighbor in original_neighbors:
        if neighbor in largest_component:
            new_adj[node].append(neighbor)
            if node not in new_adj[neighbor]:
                new_adj[neighbor].append(node)

    logger.warning(f"Articulation node {node} removed and reassigned to maintain connectivity.")
    return new_adj


def random_seed_selection(adj_list: Dict[Any, List[Any]], assigned_regions: Set[Any],
                          method: str = "gapless") -> Any:
    """
    Implements gapless random seed selection as per PRRP methodology to ensure that new regions are
    spatially contiguous.

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.
        assigned_regions (Set[Any]): A set of nodes that have already been assigned to regions.
        method (str, optional): The seed selection method. Default is "gapless".

    Returns:
        Any: The selected seed node.
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
        logger.info(f"No gapless candidate found, selected random seed: {chosen}")
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
        Dict[int, List[int]]: The adjacency list representation of the graph.
    """
    adj_list: Dict[int, List[int]] = {}
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            logger.error("METIS file is empty.")
            raise ValueError("Empty METIS file.")

        header = lines[0].strip().split()
        if len(header) < 2:
            logger.error("Invalid METIS header.")
            raise ValueError("Invalid METIS header format.")
        num_nodes = int(header[0])

        for i, line in enumerate(lines[1:], start=1):
            if line.strip():
                neighbors = [int(x) for x in line.strip().split()]
            else:
                neighbors = []
            adj_list[i] = neighbors

        for i in range(1, num_nodes + 1):
            if i not in adj_list:
                adj_list[i] = []

        logger.info(f"Loaded graph from {file_path} successfully.")
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

        logger.info(f"Graph saved to {file_path} in METIS format successfully.")
    except Exception as e:
        logger.error(f"Failed to save graph to METIS format at {file_path}: {e}")
        raise


def find_boundary_areas(region: Set[Any], adj_list: Dict[Any, List[Any]]) -> Set[Any]:
    """
    Identifies boundary areas of a given region. A boundary area is defined as a node that has at least
    one neighbor outside the region.

    Parameters:
        region (Set[Any]): A set of nodes representing the region.
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.

    Returns:
        Set[Any]: A set of nodes that are boundary areas of the region.
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
    Computes discovery and low-link values for all nodes in the graph using a DFS-based approach,
    which are essential for detecting articulation points via Tarjan's algorithm.

    Parameters:
        adj_list (Dict[Any, List[Any]]): The graph's adjacency list.

    Returns:
        Tuple[Dict[Any, int], Dict[Any, int]]:
            - disc: A dictionary mapping each node to its discovery time.
            - low: A dictionary mapping each node to its low-link value.
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


def parallel_execute(function: Callable[[Any], Any],
                     data: Iterable[Any],
                     num_threads: int = 1,
                     use_multiprocessing: bool = False) -> List[Any]:
    """
    Executes a function in parallel over the provided data if parallel processing is enabled and
    the number of threads is greater than 1. Depending on the flag 'use_multiprocessing', either
    threading or multiprocessing is used.

    Parameters:
        function (Callable[[Any], Any]): The function to execute.
        data (Iterable[Any]): An iterable of data items to process.
        num_threads (int, optional): The number of threads/processes to use for parallel processing.
                                     Defaults to 1.
        use_multiprocessing (bool, optional): If True, use multiprocessing; otherwise, use threading.
                                              Defaults to False.

    Returns:
        List[Any]: A list of results after applying the function to each data item.
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
        logger.info("Parallel processing disabled or num_threads <= 1. Executing sequentially.")
        results = [function(item) for item in data]
        logger.info("Sequential execution completed successfully.")
        return results
