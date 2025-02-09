"""
test_metis_parser.py

This module contains comprehensive unit tests for the METIS graph parser implemented in `src/metis_parser.py`.
It verifies the correctness, robustness, and error handling of `load_graph_from_metis(file_path)`.
"""

import pytest
import os
from src.metis_parser import load_graph_from_metis

# Directory for test files
TEST_DIR = "tests/data"
os.makedirs(TEST_DIR, exist_ok=True)


def write_metis_file(file_path, content):
    """Helper function to write METIS file content."""
    with open(file_path, "w") as f:
        f.write(content)

### ✅ BASIC FUNCTIONALITY TESTS ###


def test_unweighted_graph():
    """Test a simple unweighted METIS graph with correct adjacency list output."""
    file_path = os.path.join(TEST_DIR, "test_unweighted.graph")
    content = """5 4
1 2
2 3
3 4
4 5
"""
    write_metis_file(file_path, content)

    adjacency_list, num_nodes, num_edges = load_graph_from_metis(file_path)

    expected_adjacency_list = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3]
    }

    assert num_nodes == 5
    assert num_edges == 4
    assert adjacency_list == expected_adjacency_list


def test_graph_with_edge_weights():
    """Test a METIS graph with edge weights."""
    file_path = os.path.join(TEST_DIR, "test_weighted_edges.graph")
    content = """4 3 001
1 2 3
2 3 5
3 4 2
"""
    write_metis_file(file_path, content)

    adjacency_list, num_nodes, num_edges = load_graph_from_metis(file_path)

    expected_adjacency_list = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2]
    }

    assert num_nodes == 4
    assert num_edges == 3
    assert adjacency_list == expected_adjacency_list


def test_graph_with_vertex_weights():
    """Test a METIS graph with vertex weights (weights ignored in adjacency list)."""
    file_path = os.path.join(TEST_DIR, "test_vertex_weights.graph")
    content = """4 3 010
2 1 2
5 2 3
1 3 4
4 4
"""
    write_metis_file(file_path, content)

    adjacency_list, num_nodes, num_edges = load_graph_from_metis(file_path)

    expected_adjacency_list = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2]
    }

    assert num_nodes == 4
    assert num_edges == 3
    assert adjacency_list == expected_adjacency_list


def test_graph_with_vertex_and_edge_weights():
    """Test a METIS graph with both vertex and edge weights."""
    file_path = os.path.join(TEST_DIR, "test_vertex_edge_weights.graph")
    content = """4 3 011
                 2 1 2 3
                 5 2 3 5
                 1 3 4 2
                 4 4
                """
    write_metis_file(file_path, content)

    adjacency_list, num_nodes, num_edges = load_graph_from_metis(file_path)

    expected_adjacency_list = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2]
    }

    assert num_nodes == 4
    assert num_edges == 3
    assert adjacency_list == expected_adjacency_list


### ✅ ERROR HANDLING TESTS ###
def test_empty_file():
    """Test an empty METIS file should raise ValueError."""
    file_path = os.path.join(TEST_DIR, "test_empty.graph")
    write_metis_file(file_path, "")

    with pytest.raises(ValueError, match="Empty METIS file."):
        load_graph_from_metis(file_path)


def test_comment_only_file():
    """Test a METIS file with only comments should raise ValueError."""
    file_path = os.path.join(TEST_DIR, "test_comment_only.graph")
    content = """% This is a comment
% Another comment
% No actual graph data
"""
    write_metis_file(file_path, content)

    with pytest.raises(ValueError, match="Empty METIS file."):
        load_graph_from_metis(file_path)


def test_invalid_header():
    """Test a METIS file with an invalid header (missing num_nodes or num_edges)."""
    file_path = os.path.join(TEST_DIR, "test_invalid_header.graph")
    content = """5
1 2
2 3
"""
    write_metis_file(file_path, content)

    with pytest.raises(ValueError, match="Invalid METIS header"):
        load_graph_from_metis(file_path)


def test_malformed_vertex_line():
    """Test a METIS file with a malformed vertex line (incorrect number of tokens)."""
    file_path = os.path.join(TEST_DIR, "test_malformed_vertex.graph")
    content = """3 2
1
2 3
"""
    write_metis_file(file_path, content)

    with pytest.raises(ValueError, match="Insufficient vertex lines in METIS file."):
        load_graph_from_metis(file_path)


def test_odd_edge_weight_tokens():
    """Test a METIS file with an odd number of tokens for edge weights (should raise an error)."""
    file_path = os.path.join(TEST_DIR, "test_odd_edge_weights.graph")
    content = """3 2 001
1 2 3 4
2 3
"""
    write_metis_file(file_path, content)

    with pytest.raises(ValueError, match="Edge weights tokens count error in vertex"):
        load_graph_from_metis(file_path)


def test_out_of_range_neighbor():
    """Test a METIS file with an out-of-range neighbor index (should raise an error)."""
    file_path = os.path.join(TEST_DIR, "test_out_of_range_neighbor.graph")
    content = """3 2
1 4
2 3
"""
    write_metis_file(file_path, content)

    with pytest.raises(ValueError, match="Neighbor index out of range in vertex"):
        load_graph_from_metis(file_path)


### ✅ PERFORMANCE TESTS ###
def test_large_graph():
    """Test loading a large METIS graph file to ensure performance and correctness."""
    file_path = "data/PGPgiantcompo.graph"

    try:
        adjacency_list, num_nodes, num_edges = load_graph_from_metis(file_path)
        assert isinstance(adjacency_list, dict)
        assert isinstance(num_nodes, int)
        assert isinstance(num_edges, int)
        assert num_nodes > 0 and num_edges > 0
    except FileNotFoundError:
        pytest.skip("PGPgiantcompo.graph not found, skipping large graph test.")


### ✅ RUNNING THE TESTS ###
if __name__ == "__main__":
    pytest.main()
