import pytest
import random
import time
from collections import deque

# Import functions to be tested from graph_prrp
from src.graph_prrp import (
    run_graph_prrp,
    grow_partition,
    merge_disconnected_areas,
    split_partition
)
# Also import some utility functions for connectivity checking
from src.utils import (
    construct_adjacency_list,
    is_articulation_point,
    find_connected_components,
    find_boundary_areas
)

# -------------------------------
# Fixtures for reusable graph setups
# -------------------------------


@pytest.fixture
def small_graph():
    """
    Creates a simple 10-node linear graph.
    Node i connects to i-1 and i+1 (if available).
    """
    graph = {}
    for i in range(1, 11):
        neighbors = set()
        if i > 1:
            neighbors.add(i - 1)
        if i < 10:
            neighbors.add(i + 1)
        graph[i] = neighbors
    return graph


@pytest.fixture
def uneven_graph():
    """
    Creates a 17-node linear graph.
    This will force uneven partitioning when p=3.
    """
    graph = {}
    for i in range(1, 18):
        neighbors = set()
        if i > 1:
            neighbors.add(i - 1)
        if i < 17:
            neighbors.add(i + 1)
        graph[i] = neighbors
    return graph


@pytest.fixture
def disconnected_graph():
    """
    Creates a graph with two disconnected components.
    Component 1: nodes 1-3, Component 2: nodes 4-6.
    """
    return {
        1: {2},
        2: {1, 3},
        3: {2},
        4: {5},
        5: {4, 6},
        6: {5}
    }


@pytest.fixture
def cycle_graph():
    """
    Creates a cycle graph of 6 nodes: 1-2-3-4-5-6-1.
    """
    nodes = [1, 2, 3, 4, 5, 6]
    graph = {}
    for i in range(len(nodes)):
        graph[nodes[i]] = {nodes[(i - 1) % len(nodes)],
                           nodes[(i + 1) % len(nodes)]}
    return graph


@pytest.fixture
def isolated_graph():
    """
    Creates a graph with 8 nodes.
    Nodes 1-5 form a line; nodes 6-8 are isolated.
    """
    graph = {}
    for i in range(1, 6):
        neighbors = set()
        if i > 1:
            neighbors.add(i - 1)
        if i < 5:
            neighbors.add(i + 1)
        graph[i] = neighbors
    # Isolated nodes
    graph[6] = set()
    graph[7] = set()
    graph[8] = set()
    return graph


@pytest.fixture
def hub_graph():
    """
    Creates a star graph.
    Central node 1 connects to nodes 2-8.
    """
    graph = {1: set(range(2, 9))}
    for i in range(2, 9):
        graph[i] = {1}
    return graph


@pytest.fixture
def grid_graph():
    """
    Creates a 3x3 grid graph with node IDs 1 to 9.
    Each node connects to its immediate up/down/left/right neighbors.
    """
    graph = {}

    def node_id(row, col):
        return row * 3 + col + 1
    for row in range(3):
        for col in range(3):
            nid = node_id(row, col)
            neighbors = set()
            for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r = row + drow
                c = col + dcol
                if 0 <= r < 3 and 0 <= c < 3:
                    neighbors.add(node_id(r, c))
            graph[nid] = neighbors
    return graph


@pytest.fixture
def large_random_graph():
    """
    Creates a random graph with 1000 nodes.
    Each node is randomly connected to 3-5 others.
    """
    num_nodes = 1000
    graph = {i: set() for i in range(1, num_nodes + 1)}
    for i in range(1, num_nodes + 1):
        num_neighbors = random.randint(3, 5)
        possible = list(range(1, num_nodes + 1))
        possible.remove(i)
        neighbors = set(random.sample(possible, num_neighbors))
        graph[i] = neighbors
        for nbr in neighbors:
            graph[nbr].add(i)
    return graph

# -------------------------------
# Test Cases
# -------------------------------


def test_basic_partitioning(small_graph):
    """Basic Graph Partitioning Test: 10-node graph, p=2 partitions."""
    partitions = run_graph_prrp(small_graph, p=2, C=5, MR=3, MS=6)
    assert len(partitions) == 2
    # Verify that each partition is connected.
    for part in partitions.values():
        comp = find_connected_components(
            {n: list(small_graph[n] & part) for n in part})
        assert len(comp) == 1, "Partition is not connected."


def test_uneven_partitioning(uneven_graph):
    """Uneven Partitioning: 17-node graph, p=3 partitions."""
    partitions = run_graph_prrp(uneven_graph, p=3, C=6, MR=3, MS=7)
    assert len(partitions) == 3
    total_nodes = sum(len(part) for part in partitions.values())
    assert total_nodes == 17


def test_single_partition(small_graph):
    """Single Partition Test: p=1 returns all nodes in one partition."""
    partitions = run_graph_prrp(small_graph, p=1, C=10, MR=3, MS=10)
    assert len(partitions) == 1
    all_nodes = set().union(*partitions.values())
    assert all_nodes == set(small_graph.keys())


def test_graph_smaller_than_p(small_graph):
    """Graph Smaller Than p: 10 nodes but requesting p=15 should raise ValueError."""
    with pytest.raises(ValueError):
        run_graph_prrp(small_graph, p=15, C=1, MR=3, MS=2)


def test_handling_disconnected_graph(disconnected_graph):
    """Handling Disconnected Graph: Merging disconnected components."""
    partition = set(disconnected_graph.keys())
    merged = merge_disconnected_areas(disconnected_graph, set(), partition)
    comp = find_connected_components(
        {n: list(disconnected_graph[n] & merged) for n in merged})
    assert len(comp) == 1, "Merged partition is not connected."


def test_graph_with_bridges(hub_graph):
    """
    Graph With Bridges (Articulation Points):
    In a star graph the central node is an articulation point.
    When growing a partition from a peripheral node, the hub should be avoided.
    """
    graph = hub_graph.copy()
    U = set(graph.keys())
    U.discard(1)  # Remove hub for this test.
    partition = grow_partition(graph, U, p=1, c=3, MR=3)
    assert 1 not in partition, "Articulation point (hub) should be avoided in partition growth."


def test_graph_with_cycles(cycle_graph):
    """Graph With Cycles: Ensure partitioning a cycle graph results in connected partitions."""
    partitions = run_graph_prrp(cycle_graph, p=2, C=3, MR=3, MS=4)
    for part in partitions.values():
        comp = find_connected_components(
            {n: list(cycle_graph[n] & part) for n in part})
        assert len(comp) == 1, "Cycle graph partition is not connected."


def test_graph_with_isolated_nodes(isolated_graph):
    """Graph With Isolated Nodes: All nodes (even isolated) are assigned to partitions."""
    partitions = run_graph_prrp(isolated_graph, p=3, C=3, MR=3, MS=4)
    total_nodes = sum(len(part) for part in partitions.values())
    assert total_nodes == len(isolated_graph)
    for part in partitions.values():
        comp = find_connected_components(
            {n: list(isolated_graph[n] & part) for n in part})
        assert len(comp) == 1, "Partition with isolated node is not connected."


def test_graph_with_high_degree_nodes(hub_graph):
    """
    Graph With High-Degree Nodes (Hubs):
    Verify that partitions remain balanced when high-degree nodes exist.
    """
    partitions = run_graph_prrp(hub_graph, p=3, C=3, MR=3, MS=4)
    for part in partitions.values():
        comp = find_connected_components(
            {n: list(hub_graph[n] & part) for n in part})
        assert len(comp) == 1, "Partition with high-degree node is not connected."


def test_excessively_large_partition_request(small_graph):
    """
    Excessively Large Partition Request:
    For a 10-node graph, requesting a partition size (C) of 100 should raise a ValueError.
    """
    with pytest.raises(ValueError):
        run_graph_prrp(small_graph, p=2, C=100, MR=3, MS=110)


def test_boundary_nodes_in_splitting(small_graph):
    """
    Boundary Nodes in Splitting:
    Test that splitting a partition using split_partition maintains connectivity.
    """
    partition = set(small_graph.keys())
    new_parts = split_partition(small_graph, partition, ci=5)
    for part in new_parts:
        comp = find_connected_components(
            {n: list(small_graph[n] & part) for n in part})
        assert len(comp) == 1, "Split partition is not connected."
    total_nodes = sum(len(part) for part in new_parts)
    assert total_nodes == len(small_graph)


def test_randomized_graph_stress(large_random_graph):
    """
    Randomized Graph Stress Test:
    Partition a 1000-node random graph and ensure completion within a reasonable time.
    """
    start = time.time()
    partitions = run_graph_prrp(large_random_graph, p=10, C=100, MR=5, MS=150)
    duration = time.time() - start
    assert duration < 10, "Stress test exceeded time limit."
    total_nodes = sum(len(part) for part in partitions.values())
    assert total_nodes == len(large_random_graph)


def test_grid_graph_partitioning(grid_graph):
    """
    Partitioning on Grid Graph:
    Ensure that partitioning a grid graph produces even, connected partitions.
    """
    partitions = run_graph_prrp(grid_graph, p=3, C=3, MR=3, MS=5)
    for part in partitions.values():
        comp = find_connected_components(
            {n: list(grid_graph[n] & part) for n in part})
        assert len(comp) == 1, "Grid graph partition is not connected."
    total_nodes = sum(len(part) for part in partitions.values())
    assert total_nodes == len(grid_graph)


def test_star_graph_partitioning(hub_graph):
    """
    Edge Case: Star Graph:
    Verify that partitioning a star graph properly handles the central hub.
    """
    partitions = run_graph_prrp(hub_graph, p=2, C=3, MR=3, MS=4)
    for part in partitions.values():
        comp = find_connected_components(
            {n: list(hub_graph[n] & part) for n in part})
        assert len(comp) == 1, "Star graph partition is not connected."
    total_nodes = sum(len(part) for part in partitions.values())
    assert total_nodes == len(hub_graph)


def test_empty_graph():
    """
    Error Handling: Empty Graph.
    Running PRRP on an empty graph should raise a ValueError.
    """
    empty_graph = {}
    with pytest.raises(ValueError):
        run_graph_prrp(empty_graph, p=1, C=1, MR=3, MS=1)


def test_invalid_input_type():
    """
    Error Handling: Invalid Input Type.
    Passing a non-dictionary input should raise a TypeError.
    """
    with pytest.raises(TypeError):
        run_graph_prrp([1, 2, 3], p=1, C=1, MR=3, MS=1)
