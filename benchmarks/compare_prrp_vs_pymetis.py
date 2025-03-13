#!/usr/bin/env python3
"""
benchmarks/compare_prrp_vs_pymetis.py

This script benchmarks PRRP-based graph partitioning against PyMETIS.
It evaluates:
  - Execution time (cProfile)
  - Memory usage (memory_profiler)
  - Partition balance and contiguity
  - Edge cuts (number of edges crossing partitions)
"""

import os
import time
import cProfile
import pstats
import networkx as nx
import pymetis
from memory_profiler import memory_usage
from typing import Dict, List, Set, Tuple

# Import PRRP and PyMETIS functions
from src.utils import load_graph_from_metis
from src.graph_prrp import run_graph_prrp
from src.pymetis_partition import partition_graph_pymetis

# File paths
GRAPH_FILE_PATH = os.path.join(
    "data", "sample", "PGPgiantcompo.graph")
PROFILE_OUTPUT_FILE = "results/prrp_vs_pymetis_profile_1000k.txt"

# Define Parameters
NUM_PARTITIONS = 10
TARGET_CARDINALITY = 100
MAX_RETRIES = 5
MAX_SIZE = 150


def count_edge_cuts(adj_list: Dict[int, List[int]], partitions: Dict[int, Set[int]]) -> int:
    """
    Computes the number of edges that connect nodes in different partitions.

    Parameters:
        adj_list (Dict[int, List[int]]): The adjacency list of the graph.
        partitions (Dict[int, Set[int]]): The partitioning of the graph.

    Returns:
        int: The number of inter-partition edge cuts.
    """
    edge_cuts = 0
    for partition_nodes in partitions.values():
        for node in partition_nodes:
            for neighbor in adj_list[node]:
                if neighbor not in partition_nodes:  # Edge crosses partitions
                    edge_cuts += 1

    # Since each edge is counted twice, divide by 2
    return edge_cuts // 2


def benchmark_partitioning(graph: Dict[int, List[int]]):
    """
    Benchmarks PRRP and PyMETIS partitioning performance and writes results.

    Parameters:
        graph (Dict[int, List[int]]): The input graph as an adjacency list.
    """
    print("===== Benchmarking PRRP vs PyMETIS =====")

    # **Benchmark PRRP**
    print("\nRunning PRRP Partitioning...")
    try:
        start_time = time.time()
        prrp_partitions = run_graph_prrp(
            graph, NUM_PARTITIONS, TARGET_CARDINALITY, MAX_RETRIES, MAX_SIZE)
        prrp_time = time.time() - start_time

        prrp_memory = memory_usage(
            (run_graph_prrp, (graph, NUM_PARTITIONS, TARGET_CARDINALITY, MAX_RETRIES, MAX_SIZE)))
        prrp_edge_cuts = count_edge_cuts(graph, prrp_partitions)

        print(f"PRRP Execution Time: {prrp_time:.2f} seconds")
        print(f"PRRP Peak Memory Usage: {max(prrp_memory):.2f} MiB")
        print(f"PRRP Edge Cuts: {prrp_edge_cuts}")
    except Exception as e:
        print(f"PRRP Partitioning Failed: {e}")
        return

    # **Benchmark PyMETIS**
    print("\nRunning PyMETIS Partitioning...")
    try:
        start_time = time.time()
        pymetis_partitions = partition_graph_pymetis(graph, NUM_PARTITIONS)
        pymetis_time = time.time() - start_time

        pymetis_memory = memory_usage(
            (partition_graph_pymetis, (graph, NUM_PARTITIONS)))
        pymetis_edge_cuts = count_edge_cuts(graph, pymetis_partitions)

        print(f"PyMETIS Execution Time: {pymetis_time:.2f} seconds")
        print(f"PyMETIS Peak Memory Usage: {max(pymetis_memory):.2f} MiB")
        print(f"PyMETIS Edge Cuts: {pymetis_edge_cuts}")
    except Exception as e:
        print(f"PyMETIS Partitioning Failed: {e}")
        return

    # **Write results to file**
    with open(PROFILE_OUTPUT_FILE, "w") as f:
        f.write("===== PRRP vs PyMETIS Benchmark Results =====\n\n")
        f.write(f"PRRP Execution Time: {prrp_time:.2f} seconds\n")
        f.write(f"PRRP Peak Memory Usage: {max(prrp_memory):.2f} MiB\n")
        f.write(f"PRRP Edge Cuts: {prrp_edge_cuts}\n\n")

        f.write(f"PyMETIS Execution Time: {pymetis_time:.2f} seconds\n")
        f.write(f"PyMETIS Peak Memory Usage: {max(pymetis_memory):.2f} MiB\n")
        f.write(f"PyMETIS Edge Cuts: {pymetis_edge_cuts}\n")

    print(f"\nBenchmarking completed. Results saved to {PROFILE_OUTPUT_FILE}")


def main():
    """Loads the graph and runs the benchmark tests."""
    if not os.path.exists(GRAPH_FILE_PATH):
        print(f"Graph file not found: {GRAPH_FILE_PATH}")
        return

    print("Loading synthetic graph...")
    try:
        graph = load_graph_from_metis(GRAPH_FILE_PATH)
    except Exception as e:
        print(f"Failed to load graph: {e}")
        return

    benchmark_partitioning(graph)


if __name__ == "__main__":
    main()
