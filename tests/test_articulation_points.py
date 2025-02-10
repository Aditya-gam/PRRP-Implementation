"""
tests/test_articulation_points.py

Extensive tests for two key utility functions:
  1. find_articulation_points(G)
  2. construct_adjacency_list(areas)

These tests cover a wide variety of cases—from trivial graphs and edge cases to real-world METIS graph data—
to ensure the functions behave correctly under all circumstances.
"""

import os
import pytest
import geopandas as gpd
from shapely.geometry import box
from src.utils import (
    construct_adjacency_list,
    find_articulation_points,
)
from src.metis_parser import load_graph_from_metis

# ===============================
# Tests for find_articulation_points
# ===============================


def test_find_articulation_points_empty():
    """An empty graph should return an empty set of articulation points."""
    graph = {}
    aps = find_articulation_points(graph)
    assert aps == set()


def test_find_articulation_points_single_node():
    """A graph with a single node and no edges has no articulation points."""
    graph = {0: []}
    aps = find_articulation_points(graph)
    assert aps == set()


def test_find_articulation_points_triangle():
    """
    In a triangle (cycle) graph (nodes 0, 1, 2 fully connected), 
    there are no articulation points.
    """
    graph = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1]
    }
    aps = find_articulation_points(graph)
    assert aps == set()


def test_find_articulation_points_star():
    """
    In a star graph with central node 0 connected to nodes 1,2,3,4,
    the center (0) is an articulation point.
    """
    graph = {
        0: [1, 2, 3, 4],
        1: [0],
        2: [0],
        3: [0],
        4: [0]
    }
    aps = find_articulation_points(graph)
    assert aps == {0}


def test_find_articulation_points_chain():
    """
    In a chain graph 0-1-2-3, the internal nodes (1 and 2) are articulation points.
    """
    graph = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2]
    }
    aps = find_articulation_points(graph)
    assert aps == {1, 2}


def test_find_articulation_points_disconnected():
    """
    For a disconnected graph with two components:
      - Component 1: a cycle (0-1-2) with no articulation.
      - Component 2: a chain (3-4-5) where node 4 is an articulation point.
    The overall set should be {4}.
    """
    graph = {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1],
        3: [4],
        4: [3, 5],
        5: [4]
    }
    aps = find_articulation_points(graph)
    assert aps == {4}


def test_find_articulation_points_with_self_loop():
    """
    A graph with a self-loop should ignore the self-loop.
    Example: 0 has a self-loop and an edge to 1; the graph is essentially a 2-node edge.
    No articulation points are expected.
    """
    graph = {
        0: [0, 1],
        1: [0]
    }
    aps = find_articulation_points(graph)
    assert aps == set()


@pytest.mark.skipif(
    not os.path.exists(os.path.join(
        os.getcwd(), "data", "PGPgiantcompo.graph")),
    reason="Real-world graph file 'data/PGPgiantcompo.graph' not available."
)
def test_find_articulation_points_real_data():
    """
    Load the real-world METIS graph and verify that the returned articulation points 
    are a subset of the graph's nodes.
    """
    file_path = os.path.join(os.getcwd(), "data", "PGPgiantcompo.graph")
    adj, num_nodes, num_edges = load_graph_from_metis(file_path)
    aps = find_articulation_points(adj)
    # Ensure every articulation point is a node in the graph.
    assert all(node in adj for node in aps)
    assert isinstance(aps, set)


def test_load_graph_from_metis_valid(metis_file):
    """
    Test loading a valid METIS file.
    """
    adj_list, num_nodes, num_edges = load_graph_from_metis(metis_file)
    # Update expected output to reflect 0-based indexing.
    expected = {
        0: [1, 2],
        1: [0, 3],
        2: [0],
        3: [1]
    }
    assert adj_list == expected, "Loaded adjacency list does not match expected output."


# ===============================
# Tests for construct_adjacency_list
# ===============================

def test_construct_adjacency_list_dict():
    """
    When input is already a dictionary (graph-based), the function should
    convert neighbor lists to sets without altering keys.
    """
    input_dict = {
        0: [1, 2],
        1: [0],
        2: [0]
    }
    result = construct_adjacency_list(input_dict)
    assert isinstance(result, dict)
    for key, value in result.items():
        assert isinstance(value, set)
    assert result[0] == {1, 2}
    assert result[1] == {0}
    assert result[2] == {0}


def test_construct_adjacency_list_geodataframe():
    """
    Create a GeoDataFrame with three polygons:
      - Two adjacent squares (should be adjacent to each other).
      - One separate square (should have no neighbors).
    """
    # Create three boxes:
    # Box 0: from (0,0) to (1,1)
    # Box 1: from (1,0) to (2,1) -- touches box 0 along vertical edge.
    # Box 2: from (2,2) to (3,3) -- isolated.
    data = {'geometry': [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 2, 3, 3)]}
    gdf = gpd.GeoDataFrame(data)
    result = construct_adjacency_list(gdf)
    # Expect that 0 and 1 are adjacent, and 2 has no neighbors.
    assert result[0] == {1}
    assert result[1] == {0}
    assert result[2] == set()


def test_construct_adjacency_list_list_dummy():
    """
    When provided a list of dictionaries where every area has geometry None,
    the function should build a complete graph.
    """
    areas = [
        {'id': 'A', 'geometry': None},
        {'id': 'B', 'geometry': None},
        {'id': 'C', 'geometry': None},
    ]
    result = construct_adjacency_list(areas)
    expected = {
        'A': {'B', 'C'},
        'B': {'A', 'C'},
        'C': {'A', 'B'}
    }
    assert result == expected


def test_construct_adjacency_list_list_with_geometries():
    """
    When provided a list of dictionaries with valid geometries,
    the function should compute adjacencies based on spatial relationships.
    """
    from shapely.geometry import box
    areas = [
        {'id': 'X', 'geometry': box(0, 0, 1, 1)},
        {'id': 'Y', 'geometry': box(1, 0, 2, 1)},  # touches X along an edge.
        # touches Y along an edge, but not X.
        {'id': 'Z', 'geometry': box(2, 0, 3, 1)}
    ]
    result = construct_adjacency_list(areas)
    expected = {
        'X': {'Y'},
        'Y': {'X', 'Z'},
        'Z': {'Y'}
    }
    assert result == expected


def test_construct_adjacency_list_invalid_input():
    """
    Providing an unsupported type (e.g. an integer) should raise a TypeError.
    """
    with pytest.raises(TypeError):
        construct_adjacency_list(42)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(
        os.getcwd(), "data", "PGPgiantcompo.graph")),
    reason="Real-world graph file 'data/PGPgiantcompo.graph' not available."
)
def test_construct_adjacency_list_real_data():
    """
    Load a METIS graph from real data and pass the resulting dictionary to
    construct_adjacency_list(). Verify that the returned structure has the same
    keys and that each neighbor list is a set.
    """
    file_path = os.path.join(os.getcwd(), "data", "PGPgiantcompo.graph")
    adj, num_nodes, num_edges = load_graph_from_metis(file_path)
    result = construct_adjacency_list(adj)
    assert isinstance(result, dict)
    for key, value in result.items():
        assert isinstance(value, set)
    # Check that the keys match those of the input dictionary.
    assert set(result.keys()) == set(adj.keys())


def test_construct_adjacency_list_empty():
    """
    An empty list should return an empty dictionary.
    """
    result = construct_adjacency_list([])
    assert result == {}


def test_construct_adjacency_list_single_area():
    """
    A single area should have no neighbors.
    """
    areas = [{'id': 'A', 'geometry': None}]
    result = construct_adjacency_list(areas)
    assert result == {'A': set()}
    assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main()
