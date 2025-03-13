#!/usr/bin/env python3
"""
benchmark/profile_prrp.py

This script profiles the performance of the P-Regionalization through Recursive Partitioning (PRRP)
algorithm using a large synthetic graph dataset (100,000 nodes) in METIS format. The profiling covers:
  - Function execution times using cProfile and pstats.
  - Memory usage using memory_profiler.

Key functions profiled include:
  - run_graph_prrp (main PRRP function)
  - grow_partition
  - merge_disconnected_areas
  - split_partition
  - find_connected_components (from src/utils.py)

The profiling results are saved into 'profile_results.txt' for later review and optimization recommendations.

Usage:
    python benchmark/profile_prrp.py
"""


from src.utils import find_connected_components, load_graph_from_metis, construct_adjacency_list, random_seed_selection
from src.graph_prrp import run_graph_prrp, grow_partition, merge_disconnected_areas, split_partition
from memory_profiler import memory_usage
import cProfile
import pstats
import random
import os
import sys
# Add the project root (one level up) to sys.path so that the src module can be imported.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

# Import the PRRP functions from src

# Set up file paths (adjust if necessary)
GRAPH_FILE_PATH = os.path.join(
    "data", "sample", "synthetic_large_graph_100k.graph")
PROFILE_OUTPUT_FILE = "results/profile_results.txt"

# Define default PRRP parameters
# (These parameters might need to be tuned according to the dataset characteristics)
NUM_PARTITIONS = 10      # desired number of partitions
TARGET_CARDINALITY = 100  # ideal number of nodes per partition
MAX_RETRIES = 5           # maximum retries for growing a partition
MAX_SIZE = 150            # maximum allowed partition size before splitting


def profile_execution_time(graph):
    """
    Profiles the execution time of the main PRRP function (run_graph_prrp) using cProfile.

    Parameters:
        graph (dict): The input graph as an adjacency list.

    Returns:
        pstats.Stats: The pstats object containing the profiling statistics.
    """
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the main PRRP function
    partitions = run_graph_prrp(
        graph, NUM_PARTITIONS, TARGET_CARDINALITY, MAX_RETRIES, MAX_SIZE)

    profiler.disable()
    return partitions, profiler


def profile_memory_usage(func, *args, **kwargs):
    """
    Profiles the memory usage of a function call using memory_profiler.

    Parameters:
        func (callable): The function to profile.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        tuple: (result, max_memory, mem_usage) where:
            - result is the function’s return value.
            - max_memory is the peak memory usage in MiB.
            - mem_usage is a list of memory usage samples.
    """
    mem_usage, result = memory_usage(
        (func, args, kwargs), retval=True, interval=0.1, timeout=None)
    max_mem = max(mem_usage)
    return result, max_mem, mem_usage


def write_profile_results(profiler, time_stats, mem_stats):
    """
    Writes the profiling results (execution time and memory usage) into PROFILE_OUTPUT_FILE.

    Parameters:
        profiler (pstats.Stats): pstats object with execution time stats.
        time_stats (str): String representation of the time profiling stats.
        mem_stats (dict): Dictionary containing memory profiling information.
    """
    with open(PROFILE_OUTPUT_FILE, "w") as f:
        f.write("===== PRRP Execution Time Profiling =====\n\n")
        f.write(time_stats)
        f.write("\n\n===== Memory Usage Profiling =====\n\n")
        for key, value in mem_stats.items():
            f.write(f"Function: {key}\n")
            f.write(f"  Peak Memory Usage: {value['peak_memory']:.2f} MiB\n")
            f.write(f"  Memory Usage Samples: {value['samples']}\n\n")
        # Optionally, you can add recommendations based on observed values.
        f.write("===== Optimization Recommendations =====\n")
        f.write(
            "1. Investigate functions with high cumulative time (shown in the execution profile).\n")
        f.write("2. Consider optimizing inner loops and data structures (e.g., using sets or efficient libraries) where memory usage is high.\n")
        f.write("3. If memory usage is a concern, review the creation of large intermediate objects and consider in-place modifications.\n")


def main():
    # Check that the graph file exists
    if not os.path.exists(GRAPH_FILE_PATH):
        sys.exit(f"Graph file not found: {GRAPH_FILE_PATH}")

    print("Loading synthetic graph...")
    try:
        graph = load_graph_from_metis(GRAPH_FILE_PATH)
    except Exception as e:
        sys.exit(f"Failed to load graph: {e}")

    print("Profiling execution time of run_graph_prrp()...")
    partitions, profiler = profile_execution_time(graph)

    # Prepare time profiling stats string
    from io import StringIO
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
    ps.print_stats()
    time_stats = s.getvalue()

    print("Profiling memory usage of run_graph_prrp()...")
    # Profile memory usage for run_graph_prrp using memory_usage
    _, peak_memory_run, mem_usage_run = profile_memory_usage(
        run_graph_prrp, graph, NUM_PARTITIONS, TARGET_CARDINALITY, MAX_RETRIES, MAX_SIZE)

    # For demonstration, we profile one more function: construct_adjacency_list.
    from src.utils import construct_adjacency_list
    _, peak_memory_adj, mem_usage_adj = profile_memory_usage(
        construct_adjacency_list, graph)

    # Collect memory stats in a dictionary.
    mem_stats = {
        "run_graph_prrp": {"peak_memory": peak_memory_run, "samples": mem_usage_run},
        "construct_adjacency_list": {"peak_memory": peak_memory_adj, "samples": mem_usage_adj},
    }

    # Write the profiling results to file
    write_profile_results(profiler, time_stats, mem_stats)
    print(f"Profiling completed. Results saved to {PROFILE_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
