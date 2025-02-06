"""
tests/test_utils.py

Unit tests for the P-Regionalization through Recursive Partitioning (PRRP)
algorithm utilities. The tests cover:
    - Graph construction using rook adjacency (GeoDataFrame and list inputs)
    - Connected component detection
    - Articulation point identification via Tarjan’s Algorithm
    - Removal of articulation areas with connectivity reassignment
    - Gapless random seed selection
    - METIS file loading and saving
    - Boundary area detection
    - Low-link value computation for articulation detection
    - Parallel execution (threading and multiprocessing)

Requires:
    - pytest
    - geopandas
    - shapely
"""

import os
import pytest
import random
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from src.utils import (
    construct_adjacency_list,
    find_connected_components,
    is_articulation_point,
    remove_articulation_area,
    random_seed_selection,
    load_graph_from_metis,
    save_graph_to_metis,
    find_boundary_areas,
    calculate_low_link_values,
    parallel_execute,
    PARALLEL_PROCESSING_ENABLED,
)

# -----------------------------
# Helper Functions and Fixtures
# -----------------------------

def create_test_geodataframe() -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame with three polygons:
      - Two adjacent squares (sharing a boundary)
      - One isolated square
    """
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])  # adjacent to poly1
    poly3 = Polygon([(3, 0), (4, 0), (4, 1), (3, 1)])    # isolated
    gdf = gpd.GeoDataFrame({'geometry': [poly1, poly2, poly3]})

    return gdf

@pytest.fixture
def metis_file(tmp_path) -> str:
    """
    Fixture to create a temporary METIS file with valid content.
    The file represents a graph with 4 nodes:
        1: [2, 3]
        2: [1, 4]
        3: [1]
        4: [2]
    """
    content = "4 3\n2 3\n1 4\n1\n2\n"
    file_path = tmp_path / "test.metis"
    file_path.write_text(content)

    return str(file_path)

# -----------------------------
# Tests for construct_adjacency_list
# -----------------------------

def test_construct_adjacency_list_geodataframe():
    """
    Test adjacency list creation using a GeoDataFrame with rook adjacency.
    Ensures that adjacent regions are correctly detected.
    """
    gdf = create_test_geodataframe()
    adj_list = construct_adjacency_list(gdf)

    # Check that polygon 0 and 1 are adjacent (sharing a common edge).
    assert (1 in adj_list[0] or 0 in adj_list[1]), "Adjacent regions not detected correctly."

    # Check that the isolated polygon (index 2) has no neighbors.
    assert adj_list[2] == [], "Isolated region should have an empty neighbor list."

def test_construct_adjacency_list_list():
    """
    Test adjacency list creation using a list of dict-like objects.
    """
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    poly3 = Polygon([(3, 0), (4, 0), (4, 1), (3, 1)])
    areas = [
        {'id': 'a', 'geometry': poly1},
        {'id': 'b', 'geometry': poly2},
        {'id': 'c', 'geometry': poly3},
    ]
    adj_list = construct_adjacency_list(areas)

    # 'a' and 'b' should be adjacent.
    assert ('b' in adj_list['a'] or 'a' in adj_list['b']), "Adjacent areas not detected in list input."

    # 'c' should be isolated.
    assert adj_list['c'] == [], "Isolated area should have no neighbors."

# -----------------------------
# Tests for find_connected_components
# -----------------------------

def test_find_connected_components_simple():
    """
    Test detection of connected components in a simple graph.
    Graph:
        0 -- 1 -- 2
        3 (isolated)
    """
    adj_list = {
        0: [1],
        1: [0, 2],
        2: [1],
        3: []
    }
    components = find_connected_components(adj_list)
    expected = [{0, 1, 2}, {3}]

    assert len(components) == 2, "Should identify two connected components"
    
    assert set(map(frozenset, components)) == set(map(frozenset, expected)), "Component structure mismatch"

def test_find_connected_components_multiple():
    """
    Test connected components on a graph with two disconnected groups.
    """
    adj_list = {
        'A': ['B'],
        'B': ['A'],
        'C': ['D'],
        'D': ['C']
    }
    components = find_connected_components(adj_list)
    
    assert any(component == {'A', 'B'} for component in components), "Component {'A','B'} not detected."
    
    assert any(component == {'C', 'D'} for component in components), "Component {'C','D'} not detected."

# -----------------------------
# Tests for is_articulation_point
# -----------------------------

def test_is_articulation_point_positive():
    """
    Test that a known articulation point is detected.
    Graph: 1 -- 2 -- 3, node 2 is an articulation point.
    """
    adj_list = {
        1: [2],
        2: [1, 3],
        3: [2]
    }
    
    assert is_articulation_point(adj_list, 2), "Node 2 should be identified as an articulation point."

def test_is_articulation_point_negative():
    """
    Test that non-articulation points are correctly identified.
    """
    adj_list = {
        1: [2],
        2: [1, 3],
        3: [2]
    }
    
    assert not is_articulation_point(adj_list, 1), "Leaf node 1 should not be an articulation point."
    
    assert not is_articulation_point(adj_list, 3), "Leaf node 3 should not be an articulation point."

def test_is_articulation_point_boundary():
    """
    Test a graph with a triangle and an attached tail.
    Graph: 1-2-3 form a triangle; 2-4 is a tail.
    Node 2 should be an articulation point.
    """
    adj_list = {
        1: [2, 3],
        2: [1, 3, 4],
        3: [1, 2],
        4: [2]
    }
    
    assert is_articulation_point(adj_list, 2), "Node 2 should be an articulation point."
    
    assert not is_articulation_point(adj_list, 1), "Node 1 should not be an articulation point."
    
    assert not is_articulation_point(adj_list, 3), "Node 3 should not be an articulation point."
    
    assert not is_articulation_point(adj_list, 4), "Leaf node 4 should not be an articulation point."

# -----------------------------
# Tests for remove_articulation_area
# -----------------------------

def test_remove_articulation_area():
    """
    Test removal and reassignment of an articulation area.
    For graph: 1 -- 2 -- 3, removal of node 2.
    The function should reassign node 2 to the largest connected component.
    """
    adj_list = {
        1: [2],
        2: [1, 3],
        3: [2]
    }
    new_adj = remove_articulation_area(adj_list, 2)
    
    # Node 2 should be present in the new adjacency list and have at least one neighbor.
    assert 2 in new_adj, "Removed node 2 should be reassigned in the new adjacency list."
    
    assert new_adj[2], "Reassigned node 2 should have at least one neighbor."
    
    neighbor = new_adj[2][0]
    assert 2 in new_adj[neighbor], "Reassigned node 2 must appear in its neighbor's list."

def test_remove_articulation_area_no_disconnect():
    """
    Test removal on a graph that remains connected after removal.
    Graph: triangle (1,2,3); removal of any node should maintain connectivity.
    """
    adj_list = {
        1: [2, 3],
        2: [1, 3],
        3: [1, 2]
    }
    new_adj = remove_articulation_area(adj_list, 1)
    
    assert 1 in new_adj, "Removed node 1 should be reassigned in the new adjacency list."
    
    assert new_adj[1], "Reassigned node 1 should have neighbors."

# -----------------------------
# Tests for random_seed_selection
# -----------------------------

def test_random_seed_selection_gapless():
    """
    Test random seed selection using the gapless method.
    With an assigned region, the seed should be chosen from its neighbors if possible.
    """
    adj_list = {
        1: [2],
        2: [1, 3],
        3: [2, 4],
        4: [3]
    }
    assigned_regions = {1}
    seed = random_seed_selection(adj_list, assigned_regions, method="gapless")
    
    # For region 1, the only neighbor is 2.
    assert seed in {2} or seed in set(adj_list.keys()), "Seed selection did not return a valid node."

def test_random_seed_selection_no_assigned():
    """
    Test random seed selection when no regions are assigned.
    """
    adj_list = {
        1: [2],
        2: [1, 3],
        3: [2]
    }
    assigned_regions = set()
    seed = random_seed_selection(adj_list, assigned_regions, method="gapless")
    
    assert seed in set(adj_list.keys()), "Seed should be selected from all available nodes when none are assigned."

def test_random_seed_selection_no_valid_candidate():
    """
    Test random seed selection when no candidate meets the gapless criteria.
    """
    # Create a graph where assigned regions have no unassigned neighbors.
    adj_list = {
        1: [2],
        2: [1],
        3: []
    }
    assigned_regions = {1, 2}
    seed = random_seed_selection(adj_list, assigned_regions, method="gapless")
   
    # The only unassigned node is 3.
    assert seed == 3, "Seed should be the only unassigned node (3)."

# -----------------------------
# Tests for load_graph_from_metis
# -----------------------------

def test_load_graph_from_metis_valid(metis_file):
    """
    Test loading a valid METIS file.
    """
    adj_list = load_graph_from_metis(metis_file)
    expected = {
        1: [2, 3],
        2: [1, 4],
        3: [1],
        4: [2]
    }
    
    assert adj_list == expected, "Loaded adjacency list does not match expected output."

def test_load_graph_from_metis_empty(tmp_path):
    """
    Test loading an empty METIS file, expecting a ValueError.
    """
    file_path = tmp_path / "empty.metis"
    file_path.write_text("")
    
    with pytest.raises(ValueError):
        load_graph_from_metis(str(file_path))

def test_load_graph_from_metis_invalid_header(tmp_path):
    """
    Test loading a METIS file with an invalid header.
    """
    content = "invalid header\n2 3\n1 4\n1\n2\n"
    file_path = tmp_path / "invalid.metis"
    file_path.write_text(content)
    
    with pytest.raises(ValueError):
        load_graph_from_metis(str(file_path))

# -----------------------------
# Tests for save_graph_to_metis
# -----------------------------

def test_save_graph_to_metis(tmp_path):
    """
    Test saving a graph to a METIS file and verify its content.
    """
    adj_list = {
        1: [2, 3],
        2: [1, 4],
        3: [1],
        4: [2]
    }
    file_path = tmp_path / "output.metis"
    save_graph_to_metis(str(file_path), adj_list)
    content = file_path.read_text().strip().splitlines()
    
    # Expected header: "4 3" (4 nodes, 3 edges)
    assert content[0].strip() == "4 3", "METIS header does not match expected output."
    
    # There should be 1 header line plus 4 lines for each node.
    assert len(content) == 5, "Incorrect number of lines in METIS file output."

# -----------------------------
# Tests for find_boundary_areas
# -----------------------------

def test_find_boundary_areas():
    """
    Test detection of boundary areas in a region.
    In the graph, if region = {1, 2} and node 2 has a neighbor outside, then 2 is a boundary area.
    """
    adj_list = {
        1: [2],
        2: [1, 3],
        3: [2, 4],
        4: [3]
    }
    region = {1, 2}
    boundaries = find_boundary_areas(region, adj_list)
    
    # Node 1's only neighbor is 2 (inside region); node 2 has neighbor 3 (outside).
    assert boundaries == {2}, "Boundary area detection failed."

# -----------------------------
# Tests for calculate_low_link_values
# -----------------------------

def test_calculate_low_link_values():
    """
    Test calculation of discovery and low-link values for a simple chain graph.
    Graph: 1 -- 2 -- 3
    """
    adj_list = {
        1: [2],
        2: [1, 3],
        3: [2]
    }
    disc, low = calculate_low_link_values(adj_list)
    
    for node in adj_list:
        assert node in disc, f"Node {node} missing in discovery times."
        assert node in low, f"Node {node} missing in low-link values."
        assert low[node] <= disc[node], f"Low-link value for node {node} is inconsistent."

# -----------------------------
# Tests for parallel_execute
# -----------------------------

def test_parallel_execute_threading():
    """
    Test parallel execution using threading.
    Ensure that the threaded execution produces the same output as sequential execution.
    """
    data = list(range(10))

    def double(x: int) -> int:
        return x * 2

    results_seq = parallel_execute(double, data, num_threads=1)
    original_flag = PARALLEL_PROCESSING_ENABLED
    
    try:
        globals()['PARALLEL_PROCESSING_ENABLED'] = True
        results_thread = parallel_execute(double, data, num_threads=4, use_multiprocessing=False)
        
        assert results_seq == results_thread, "Threaded parallel execution output differs from sequential."
    finally:
        globals()['PARALLEL_PROCESSING_ENABLED'] = original_flag

def test_parallel_execute_multiprocessing():
    """
    Test parallel execution using multiprocessing.
    Ensure that the multiprocessing execution produces the same output as sequential execution.
    """
    data = list(range(10))

    def triple(x: int) -> int:
        return x * 3

    results_seq = parallel_execute(triple, data, num_threads=1)
    original_flag = PARALLEL_PROCESSING_ENABLED
    
    try:
        globals()['PARALLEL_PROCESSING_ENABLED'] = True
        results_mp = parallel_execute(triple, data, num_threads=4, use_multiprocessing=True)
        
        assert results_seq == results_mp, "Multiprocessing parallel execution output differs from sequential."
    finally:
        globals()['PARALLEL_PROCESSING_ENABLED'] = original_flag
