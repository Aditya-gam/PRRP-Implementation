#!/usr/bin/env python3
"""
run_experiments.py

This script runs experiments for both spatial and graph partitioning using the PRRP algorithm.
It executes:
  - Spatial PRRP on a shapefile dataset.
  - Graph PRRP on multiple METIS graph files.
  
For each experiment, the script measures execution time and outputs a summary
of parameters and results (e.g., partition sizes, number of nodes in each partition).

The output is saved in a text file in the directory:
  PRRP-Implementation/results/experiment_results.txt

Usage:
  python run_experiments.py
"""

import os
import time
import random
import logging
from datetime import datetime

# Import spatial PRRP functions and data loader
from temp.spatial_prrp import run_prrp as run_spatial_prrp, run_parallel_prrp
from src.prrp_data_loader import load_shapefile

# Import graph PRRP functions and METIS parser
from src.graph_prrp import run_graph_prrp
from src.metis_parser import load_graph_from_metis

# Set up logging for the experiment script
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("experiment")

# Define paths for datasets and results output
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
RESULTS_FILE = os.path.join(RESULTS_DIR, "experiment_results.txt")

# Define dataset paths
# Spatial dataset (shapefile)
SPATIAL_SHAPEFILE = os.path.join(
    BASE_DIR, "data", "cb_2015_42_tract_500k", "cb_2015_42_tract_500k.shp")
# Graph datasets (METIS format)
GRAPH_DATASETS = {
    "PGPgiantcompo": os.path.join(BASE_DIR, "data", "sample", "PGPgiantcompo.graph"),
    "synthetic_100k": os.path.join(BASE_DIR, "data", "sample", "synthetic_large_graph_100k.graph"),
    "synthetic_1000k": os.path.join(BASE_DIR, "data", "sample", "synthetic_large_graph_1000k.graph")
}


def write_results(output_str: str):
    """Append output string to the results file."""
    with open(RESULTS_FILE, "a") as f:
        f.write(output_str + "\n")


def run_spatial_experiment():
    """
    Runs the spatial PRRP experiment on the shapefile dataset.
    The experiment loads the shapefile, sets up parameters, runs the algorithm,
    and returns a summary string.
    """
    output = []
    output.append("=== Spatial PRRP Experiment ===")
    output.append(f"Dataset: {SPATIAL_SHAPEFILE}")

    areas = load_shapefile(SPATIAL_SHAPEFILE)
    if areas is None:
        msg = "Error loading shapefile."
        logger.error(msg)
        return msg

    num_areas = len(areas)
    output.append(f"Number of areas: {num_areas}")

    num_regions = 5
    target_card = num_areas // num_regions
    cardinalities = [target_card] * num_regions
    cardinalities[-1] += (num_areas - target_card * num_regions)
    output.append(f"Number of regions: {num_regions}")
    output.append(f"Target cardinalities: {cardinalities}")

    start_time = time.time()
    try:
        solutions = run_spatial_prrp(
            areas, num_regions, cardinalities)
    except Exception as e:
        msg = f"Error during spatial PRRP execution: {e}"
        logger.error(msg)
        return msg
    if not solutions:
        msg = "No valid spatial PRRP solution generated."
        logger.error(msg)
        return msg
    solution = solutions[0]
    end_time = time.time()
    exec_time = end_time - start_time

    output.append(f"Execution Time: {exec_time:.2f} seconds")
    partition_sizes = {i+1: len(region) for i, region in enumerate(solution)}
    output.append("Partition sizes:")
    for pid, size in partition_sizes.items():
        output.append(f"  Partition {pid}: {size} areas")

    return "\n".join(output)


def run_graph_experiment(dataset_name: str, file_path: str, p: int, MR: int):
    """
    Runs the graph PRRP experiment on a given METIS graph dataset.
    Parameters:
      dataset_name (str): A name/label for the dataset.
      file_path (str): Path to the METIS graph file.
      p (int): Desired number of partitions.
      MR (int): Maximum number of retries for partition growth.
    Returns:
      A summary string of the experiment results.
    """
    output = []
    output.append("=== Graph PRRP Experiment ===")
    output.append(f"Dataset: {dataset_name} ({file_path})")

    try:
        adj_list, num_nodes, num_edges = load_graph_from_metis(file_path)
    except Exception as e:
        msg = f"Error loading graph dataset '{dataset_name}': {e}"
        logger.error(msg)
        return msg

    output.append(f"Number of nodes: {num_nodes}")
    output.append(f"Number of edges: {num_edges}")

    # Choose parameters for graph partitioning.
    # For example, set target cardinality as approx. 10% of total nodes.
    C = max(1, num_nodes // 10)
    # Set maximum partition size (MS) slightly larger than target cardinality.
    MS = C + 20

    output.append(f"Parameters: p = {p}, C = {C}, MR = {MR}, MS = {MS}")

    start_time = time.time()
    try:
        partitions = run_graph_prrp(adj_list, p, C, MR, MS)
    except Exception as e:
        msg = f"Error during graph PRRP execution on '{dataset_name}': {e}"
        logger.error(msg)
        return msg
    end_time = time.time()
    exec_time = end_time - start_time

    output.append(f"Execution Time: {exec_time:.2f} seconds")
    # Report partition sizes.
    partition_sizes = {pid: len(part) for pid, part in partitions.items()}
    output.append("Partition sizes:")
    for pid, size in partition_sizes.items():
        output.append(f"  Partition {pid}: {size} nodes")

    total_assigned = sum(partition_sizes.values())
    output.append(f"Total nodes in partitions: {total_assigned}")

    return "\n".join(output)


def main():
    # Clear previous results file if exists.
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    header = "=== PRRP Experiment Results ==="
    header += f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += "=" * 40
    write_results(header)

    # Run spatial PRRP experiment
    logger.info("Running spatial PRRP experiment...")
    spatial_results = run_spatial_experiment()
    write_results(spatial_results)
    write_results("-" * 40)

    # Run graph PRRP experiments for each graph dataset.
    logger.info("Running graph PRRP experiments...")
    # Set common graph parameters (these can be tuned as needed)
    graph_p = 10
    graph_MR = 5

    for dataset_name, file_path in GRAPH_DATASETS.items():
        logger.info(f"Running experiment for dataset: {dataset_name}")
        graph_result = run_graph_experiment(
            dataset_name, file_path, graph_p, graph_MR)
        write_results(graph_result)
        write_results("-" * 40)

    footer = "=== End of PRRP Experiment Results ==="
    write_results(footer)
    logger.info(f"Experiment results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
