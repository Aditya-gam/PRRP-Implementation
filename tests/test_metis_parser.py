"""
tests/test_metis_parser.py

Unit tests for the METIS graph parser implemented in src/metis_parser.py.
These tests cover:
  - Correct adjacency list generation for unweighted graphs.
  - Handling of graphs with edge weights.
  - Handling of graphs with vertex weights.
  - Handling of graphs with both vertex and edge weights.
  - Detection and reporting of malformed files (empty files, only comments, malformed header,
    insufficient vertex lines, odd token counts for weighted edges, out-of-range neighbor indices).
  - Verification of 1-based to 0-based indexing conversion and self-loop prevention.
  - A performance test for a large METIS graph (data/PGPgiantcompo.graph).

Run the tests with:
    PYTHONPATH=$(pwd) pytest tests/test_metis_parser.py --maxfail=1 --disable-warnings -v
"""

import os
import pytest
from src.metis_parser import load_graph_from_metis

# Helper function to write graph content to a temporary file.


def write_temp_graph(tmp_path, content: str) -> str:
    file_path = tmp_path / "graph.metis"
    file_path.write_text(content)
    return str(file_path)

# ---------- Basic Functionality Tests ----------


def test_load_unweighted_graph(tmp_path):
    """
    Test a simple unweighted METIS graph.
    Graph:
      Vertex 1: neighbor 2
      Vertex 2: neighbors 1 and 3
      Vertex 3: neighbor 2
    Header "3 2" with 2 edges (each edge appears twice in the file).
    """
    content = (
        "3 2\n"
        "2\n"
        "1 3\n"
        "2\n"
    )
    file_path = write_temp_graph(tmp_path, content)
    adj_list, num_nodes, num_edges = load_graph_from_metis(file_path)
    expected_adj = {0: [1], 1: [0, 2], 2: [1]}
    assert adj_list == expected_adj
    assert num_nodes == 3
    assert num_edges == 2


def test_load_edge_weighted_graph(tmp_path):
    """
    Test a METIS graph with edge weights.
    Header "3 2 01": no vertex weights; edge weights present.
    Each vertex line has pairs: (neighbor, weight) (weights are ignored).
    """
    content = (
        "3 2 01\n"
        "2 10\n"
        "1 10 3 20\n"
        "2 20\n"
    )
    file_path = write_temp_graph(tmp_path, content)
    adj_list, num_nodes, num_edges = load_graph_from_metis(file_path)
    expected_adj = {0: [1], 1: [0, 2], 2: [1]}
    assert adj_list == expected_adj
    assert num_nodes == 3
    assert num_edges == 2


def test_load_vertex_weighted_graph(tmp_path):
    """
    Test a METIS graph with vertex weights (vertex weights are ignored in the adjacency list).
    Header "3 2 10": vertex weights present; no edge weights.
    Each vertex line: first token is vertex weight, then neighbor list.
    """
    content = (
        "3 2 10\n"
        "5 2\n"
        "3 1 3\n"
        "4 2\n"
    )
    file_path = write_temp_graph(tmp_path, content)
    adj_list, num_nodes, num_edges = load_graph_from_metis(file_path)
    expected_adj = {0: [1], 1: [0, 2], 2: [1]}
    assert adj_list == expected_adj
    assert num_nodes == 3
    assert num_edges == 2


def test_load_both_vertex_and_edge_weighted_graph(tmp_path):
    """
    Test a METIS graph with both vertex and edge weights.
    Header "3 2 11 1": vertex weights present, edge weights present, ncon = 1.
    Each vertex line: first token is vertex weight, then pairs (neighbor, edge weight).
    """
    content = (
        "3 2 11 1\n"
        "5 2 10\n"
        "3 1 10 3 20\n"
        "4 2 20\n"
    )
    file_path = write_temp_graph(tmp_path, content)
    adj_list, num_nodes, num_edges = load_graph_from_metis(file_path)
    expected_adj = {0: [1], 1: [0, 2], 2: [1]}
    assert adj_list == expected_adj
    assert num_nodes == 3
    assert num_edges == 2

# ---------- Error Handling Tests ----------


def test_empty_file(tmp_path):
    """Test that an empty METIS file raises a ValueError."""
    content = ""
    file_path = write_temp_graph(tmp_path, content)
    with pytest.raises(ValueError, match="Empty METIS file"):
        load_graph_from_metis(file_path)


def test_file_with_only_comments(tmp_path):
    """Test that a METIS file with only comment lines raises a ValueError."""
    content = (
        "% This is a comment\n"
        "% Another comment\n"
        "   % Indented comment\n"
    )
    file_path = write_temp_graph(tmp_path, content)
    with pytest.raises(ValueError, match="Empty METIS file"):
        load_graph_from_metis(file_path)


def test_malformed_header(tmp_path):
    """Test that a METIS file with an invalid header raises a ValueError."""
    content = (
        "3\n"
        "2\n"
        "1 3\n"
        "2\n"
    )
    file_path = write_temp_graph(tmp_path, content)
    with pytest.raises(ValueError, match="Invalid METIS header"):
        load_graph_from_metis(file_path)


def test_insufficient_vertex_lines(tmp_path):
    """Test that a METIS file with insufficient vertex lines raises a ValueError."""
    content = (
        "3 2\n"
        "2\n"
        "1 3\n"
    )
    file_path = write_temp_graph(tmp_path, content)
    with pytest.raises(ValueError, match="Insufficient vertex lines"):
        load_graph_from_metis(file_path)


def test_malformed_vertex_line_missing_edge_weight(tmp_path):
    """
    Test that a METIS file with an odd number of tokens for edge weights
    raises a ValueError.
    """
    content = (
        "3 2 01\n"
        "2 10\n"
        "1 3 20\n"  # This line has 3 tokens (should be even)
        "2 20\n"
    )
    file_path = write_temp_graph(tmp_path, content)
    with pytest.raises(ValueError, match="Edge weights tokens count error in vertex"):
        load_graph_from_metis(file_path)


def test_out_of_range_neighbor(tmp_path):
    """Test that a METIS file with an out-of-range neighbor index raises a ValueError."""
    content = (
        "3 2\n"
        "1 4\n"  # For vertex 1, neighbor '4' is invalid (graph has 3 nodes)
        "1 3\n"
        "2\n"
    )
    file_path = write_temp_graph(tmp_path, content)
    with pytest.raises(ValueError, match="Neighbor index out of range in vertex"):
        load_graph_from_metis(file_path)

# ---------- Indexing & Validation Tests ----------


def test_self_loop_not_included(tmp_path):
    """
    Test that a vertex line with a self-loop does not include the self-loop in the adjacency list.
    For example, for vertex 1 with line "1 2", the self-loop (1) is ignored; only neighbor 2 is included.
    """
    content = (
        "3 2\n"
        "1 2\n"  # Vertex 1: self-loop (1) ignored, valid neighbor 2 included.
        "1 3\n"
        "2\n"
    )
    file_path = write_temp_graph(tmp_path, content)
    adj_list, num_nodes, num_edges = load_graph_from_metis(file_path)
    expected_adj = {0: [1], 1: [0, 2], 2: [1]}
    assert adj_list == expected_adj


def test_edge_count_validation(tmp_path):
    """
    Test that if the header edge count does not match the computed neighbor entries,
    the parser computes the edge count as half the total neighbor entries.
    """
    content = (
        "3 1\n"
        "2\n"
        "1 3\n"
        "2\n"
    )
    file_path = write_temp_graph(tmp_path, content)
    adj_list, num_nodes, num_edges = load_graph_from_metis(file_path)
    # Total neighbor entries = 1 (vertex 1) + 2 (vertex 2) + 1 (vertex 3) = 4.
    # Computed edge count = 4 // 2 = 2.
    assert num_edges == 2

# ---------- Performance / Large Graph Test ----------


def test_large_graph_performance(tmp_path):
    """
    Attempts to load the large METIS graph (PGPgiantcompo.graph) from the data directory.
    If the file is not available, the test is skipped.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph_path = os.path.join(base_dir, "data", "PGPgiantcompo.graph")
    if not os.path.exists(graph_path):
        pytest.skip(
            "Large graph file 'PGPgiantcompo.graph' not available in data directory.")

    adj_list, num_nodes, num_edges = load_graph_from_metis(graph_path)
    assert isinstance(adj_list, dict)
    assert isinstance(num_nodes, int)
    assert isinstance(num_edges, int)
    # For a large graph, we expect a significant number of nodes.
    assert num_nodes > 1000


def test_load_pgpgiantcompo_graph():
    """
    Test loading the large METIS graph file 'PGPgiantcompo.graph' from the data directory.
    This ensures that the graph is correctly parsed and structured for use in Graph PRRP.
    If the file is missing, the test is skipped.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph_path = os.path.join(base_dir, "data", "PGPgiantcompo.graph")

    if not os.path.exists(graph_path):
        pytest.skip("PGPgiantcompo.graph not found, skipping test.")

    adj_list, num_nodes, num_edges = load_graph_from_metis(graph_path)

    # Validate the parsed data structure
    assert isinstance(adj_list, dict), "Adjacency list should be a dictionary."
    assert isinstance(
        num_nodes, int) and num_nodes > 0, "Number of nodes should be positive."
    assert isinstance(
        num_edges, int) and num_edges > 0, "Number of edges should be positive."

    # Ensure at least some nodes have neighbors (real-world connectivity)
    non_empty_nodes = sum(1 for neighbors in adj_list.values() if neighbors)
    assert non_empty_nodes > 0, "At least some nodes should have neighbors."

    print(
        f"Successfully loaded PGPgiantcompo.graph with {num_nodes} nodes and {num_edges} edges.")


# ---------- Optional: Allow Running Tests Directly ----------
if __name__ == "__main__":
    pytest.main()
