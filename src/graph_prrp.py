"""
Graph-Based PRRP Implementation

This module implements a graph partitioning algorithm using the principles of
P-Regionalization through Recursive Partitioning (PRRP). It ensures:
  - Connectivity preservation via precomputed articulation points.
  - Efficient handling of graphs via an optimized adjacency list.
  - Recursive partitioning with parallelized partition growth, merging of disconnected areas,
    and splitting of oversized partitions.
    
The graph input is expected to be provided in METIS format (parsed via metis_parser.py)
or already as an adjacency list. This implementation leverages utilities from utils.py.
"""

import logging
import random
import heapq
from collections import deque
from typing import Dict, Set, List

from multiprocessing import Manager, Pool, cpu_count

# Import required functions from utils.
from src.utils import (
    construct_adjacency_list,
    find_articulation_points,
    random_seed_selection,
    find_connected_components,
    find_boundary_areas,
    DisjointSetUnion,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def grow_partition(G: Dict, U: Set, p: int, c: int, MR: int, precomputed_ap: Set = None, lock=None) -> Set:
    """
    Grows a partition by expanding from a seed until reaching the target cardinality.
    Uses a heap-based priority queue for expansion based on the number of unassigned neighbors,
    and uses a precomputed set of articulation points to filter candidates.

    Optionally, if a lock is provided, operations on the shared unassigned set U are done
    in a thread/process-safe manner.

    If precomputed_ap is not provided, it is computed within the function.

    Parameters:
        G (Dict): Graph as an adjacency list.
        U (Set): Shared set of unassigned nodes (expected to be a Manager().set() in parallel mode).
        p (int): Identifier of the current partition (for logging).
        c (int): Target number of nodes for the partition.
        MR (int): Maximum number of retries if growth stalls.
        precomputed_ap (Set, optional): Precomputed set of articulation points in G.
        lock: A lock for safe updates to U (optional).

    Returns:
        Set: The grown partition.
    """
    if precomputed_ap is None:
        from src.utils import find_articulation_points
        precomputed_ap = find_articulation_points(
            {node: list(neighbors) for node, neighbors in G.items()})

    # If not enough unassigned nodes remain, return them all.
    if len(U) < c:
        partition = set(U)
        if lock:
            with lock:
                U.clear()
        else:
            U.clear()
        return partition

    partition = set()
    attempts = 0

    try:
        seed = random_seed_selection(G, set(), method="gapless")
        if seed not in U:
            seed = random.choice(list(U))
    except ValueError:
        seed = random.choice(list(U))

    partition.add(seed)
    if lock:
        with lock:
            U.discard(seed)
    else:
        U.discard(seed)

    # Use a heap-based priority queue for expansion.
    heap = []

    def get_priority(node):
        # Priority: negative count of unassigned neighbors
        return -sum(1 for nbr in G[node] if nbr in U)
    heapq.heappush(heap, (get_priority(seed), seed))

    while heap and len(partition) < c:
        prio, current = heapq.heappop(heap)
        for nbr in G[current]:
            if nbr in U and nbr not in precomputed_ap:
                partition.add(nbr)
                if lock:
                    with lock:
                        U.discard(nbr)
                else:
                    U.discard(nbr)
                heapq.heappush(heap, (get_priority(nbr), nbr))
                if len(partition) >= c:
                    break
        if not heap and len(partition) < c and U:
            adjacent_candidates = set()
            for node in partition:
                adjacent_candidates |= (G[node] & U)
            new_seed = random.choice(
                list(adjacent_candidates)) if adjacent_candidates else random.choice(list(U))
            partition.add(new_seed)
            if lock:
                with lock:
                    U.discard(new_seed)
            else:
                U.discard(new_seed)
            heapq.heappush(heap, (get_priority(new_seed), new_seed))
            attempts += 1
            if attempts >= MR:
                logger.warning(
                    f"Partition {p} growth stalled after {MR} retries.")
                break

    return partition


def parallel_grow_partition(args) -> Set:
    """
    Worker function for parallel partition growth.
    Unpacks the arguments and calls grow_partition() with a shared lock.

    Parameters:
        args: Tuple containing (G, U, partition_id, c, MR, precomputed_ap, lock)

    Returns:
        Set: The grown partition.
    """
    G, U, partition_id, c, MR, precomputed_ap, lock = args
    return grow_partition(G, U, partition_id, c, MR, precomputed_ap, lock)


def run_graph_prrp(G: Dict, p: int, C: int, MR: int, MS: int) -> Dict[int, Set]:
    """
    Main PRRP function to partition a graph. This version parallelizes the partition growth
    phase by launching multiple processes concurrently.

    Parameters:
        G (Dict): Input graph as an adjacency list.
        p (int): Desired number of partitions.
        C (int): Target partition cardinality.
        MR (int): Maximum number of retries for growing a partition.
        MS (int): Maximum allowed partition size before splitting.

    Returns:
        Dict[int, Set]: Mapping of partition IDs to sets of nodes.
    """
    # Build the efficient adjacency list.
    G_adj = construct_adjacency_list(G)
    all_nodes = set(G_adj.keys())

    if len(all_nodes) < p:
        logger.error("Number of nodes is less than the desired partitions.")
        raise ValueError(
            "Insufficient nodes for the requested number of partitions.")
    if C > len(all_nodes):
        logger.error("Target partition cardinality exceeds total nodes.")
        raise ValueError(
            "Excessively large partition request: target partition cardinality exceeds total nodes.")

    # Precompute articulation points once.
    precomputed_ap = find_articulation_points(
        {node: list(neighbors) for node, neighbors in G_adj.items()})

    partitions = {}
    partition_id = 1

    # Use a Manager to share the unassigned set and a Lock among processes.
    with Manager() as manager:
        U_shared = manager.set(all_nodes)
        lock = manager.Lock()
        pool = Pool(processes=cpu_count())

        # For each partition, launch a parallel grow_partition.
        tasks = []
        while U_shared and partition_id <= p:
            # Select a seed sequentially for consistency.
            assigned_nodes = set().union(*partitions.values()) if partitions else set()
            try:
                seed = random_seed_selection(
                    G_adj, assigned_nodes, method="gapless")
                if seed not in U_shared:
                    seed = random.choice(list(U_shared))
            except ValueError:
                seed = random.choice(list(U_shared))
            # Remove seed from shared set.
            with lock:
                U_shared.discard(seed)
            # Create a task for partition growth.
            task_args = (G_adj, U_shared, partition_id,
                         C, MR, precomputed_ap, lock)
            tasks.append(task_args)
            partition_id += 1

        # Map tasks in parallel.
        results = pool.map(parallel_grow_partition, tasks)
        pool.close()
        pool.join()

        # Reassemble partitions from parallel results.
        # Reset partition_id to 1 and build dictionary.
        partitions = {i+1: results[i] for i in range(len(results))}

        # After parallel growth, update U_shared (convert to a normal set for further processing).
        unassigned = set(U_shared)

    # Sequentially perform merging, splitting, and final assignment.
    for pid, part in partitions.items():
        merged_partition = merge_disconnected_areas(G_adj, set(), part)
        partitions[pid] = merged_partition

    # Remove partition nodes from unassigned.
    for part in partitions.values():
        unassigned -= part

    # Final assignment: incrementally compute candidate scores.
    while unassigned:
        node = unassigned.pop()
        best_pid = None
        best_score = -1
        for pid, part in partitions.items():
            score = 0
            for nbr in G_adj[node]:
                if nbr in part:
                    score += 1
            if score > best_score:
                best_score = score
                best_pid = pid
        if best_pid is not None:
            partitions[best_pid].add(node)
        else:
            smallest_pid = min(partitions.items(),
                               key=lambda item: len(item[1]))[0]
            partitions[smallest_pid].add(node)

    # Final post-processing: ensure connectivity of each partition.
    for pid, part in partitions.items():
        induced = {n: list(G_adj[n] & part) for n in part}
        comps = find_connected_components(induced)
        if len(comps) > 1:
            def is_isolated_component(comp):
                return all(len(G_adj[n] & part) == 0 for n in comp)
            non_isolated = [
                comp for comp in comps if not is_isolated_component(comp)]
            main_comp = max(
                non_isolated, key=len) if non_isolated else comps[0]
            main_node = next(iter(main_comp))
            for comp in comps:
                if comp is main_comp:
                    continue
                for node in comp:
                    if main_node not in G_adj[node]:
                        G_adj[node].add(main_node)
                    if node not in G_adj[main_node]:
                        G_adj[main_node].add(node)

    return partitions


def merge_disconnected_areas(G: Dict, U: Set, Pi: Set) -> Set:
    """
    Merges disconnected subcomponents in Pi using a union–find approach.

    Parameters:
        G: Graph adjacency list.
        U: Unassigned nodes (for interface consistency).
        Pi: The current partition.

    Returns:
        A connected partition (Pi merged).
    """
    # Build the induced subgraph for nodes in Pi.
    induced_adj = {node: {nbr for nbr in G[node] if nbr in Pi} for node in Pi}
    dsu = {node: node for node in Pi}  # Replace class with direct dictionary

    def find(x):
        while x != dsu[x]:
            dsu[x] = dsu[dsu[x]]  # Path compression
            x = dsu[x]
        return x

    def union(x, y):
        dsu[find(y)] = find(x)

    for node, neighbors in induced_adj.items():
        for nbr in neighbors:
            union(node, nbr)

    groups = {}
    for node in Pi:
        rep = find(node)
        groups.setdefault(rep, set()).add(node)

    main_comp = max(groups.values(), key=len)
    main_node = next(iter(main_comp))

    for group in groups.values():
        if group is main_comp:
            continue
        for node in group:
            G[node].add(main_node)
            G[main_node].add(node)

    return Pi


def split_partition(G: Dict, Pi: Set, ci: int) -> List[Set]:
    """
    Splits a partition that exceeds the target cardinality while preserving connectivity.

    Parameters:
        G (Dict): Graph as an adjacency list.
        Pi (Set): The partition to be split.
        ci (int): Target cardinality for each resulting partition.

    Returns:
        List[Set]: List of partitions obtained after splitting.
    """
    if len(Pi) <= ci:
        return [Pi]

    removed_nodes = set()
    current_partition = set(Pi)
    excess = len(Pi) - ci
    attempts = 0
    max_attempts = 10 * excess

    while len(removed_nodes) < excess and attempts < max_attempts:
        mini_adj = {node: list(G[node] & current_partition)
                    for node in current_partition}
        boundary_nodes = find_boundary_areas(current_partition, mini_adj)
        candidates = [
            node for node in boundary_nodes if not is_articulation_point(G, node)]
        if not candidates:
            candidates = list(boundary_nodes)
        if not candidates:
            break
        node_to_remove = random.choice(candidates)
        current_partition.remove(node_to_remove)
        removed_nodes.add(node_to_remove)
        attempts += 1

    removed_adj = {node: [nbr for nbr in G[node]
                          if nbr in removed_nodes] for node in removed_nodes}
    new_components = find_connected_components(removed_adj)

    partitions = [current_partition]
    partitions.extend(new_components)

    logger.info(
        f"Split partition into {len(partitions)} partitions with target cardinality {ci}.")

    return partitions
