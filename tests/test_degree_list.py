import pytest
from src.utils import compute_degree_list


def test_basic_graph():
    """
    Test a simple graph with uniform degrees where no parent-child relationships are expected.
    """
    G = {1: [2, 3], 2: [1, 4], 3: [1, 4], 4: [2, 3]}
    # All nodes have degree 2; median for each equals 2, so no node qualifies as parent.
    expected = {1: [], 2: [], 3: [], 4: []}
    result = compute_degree_list(G)
    assert result == expected


def test_nested_structure():
    """
    Test a graph where certain nodes have higher degree than the median of their neighbors.
    Parent nodes should have lower-degree neighbors assigned as children.
    """
    G = {1: [2, 3, 4],
         2: [1, 5, 6],
         3: [1],
         4: [1],
         5: [2],
         6: [2]}
    # For node 1, neighbors [2,3,4] have degrees [3,1,1] -> median is 1, so 1 qualifies as parent.
    # For node 2, neighbors [1,5,6] have degrees [3,1,1] -> median is 1, so 2 qualifies as parent.
    # Expected assignments: node 1 gets children 3 and 4; node 2 gets children 5 and 6.
    expected = {1: [3, 4], 2: [5, 6], 3: [], 4: [], 5: [], 6: []}
    result = compute_degree_list(G)
    # Compare sorted lists for consistency (order may differ)
    for k in expected:
        assert sorted(result[k]) == sorted(expected[k])
    assert set(result.keys()) == set(expected.keys())


def test_isolated_nodes():
    """
    Test graphs that contain isolated nodes (with no neighbors).
    Isolated nodes should remain without any parent-child relationships.
    """
    G = {1: [],
         2: [3],
         3: [2],
         4: []}
    expected = {1: [], 2: [], 3: [], 4: []}
    result = compute_degree_list(G)
    assert result == expected


def test_disconnected_components():
    """
    Test a graph with two disconnected components.
    Each connected component should be processed independently.
    """
    G = {1: [2],
         2: [1],
         3: [4, 5],
         4: [3],
         5: [3]}
    # Component 1: nodes 1 and 2 have degree 1 each; no parent qualifies.
    # Component 2: node 3 has degree 2 with neighbors (degree 1 each) -> qualifies as parent.
    expected = {1: [], 2: [], 3: [4, 5], 4: [], 5: []}
    result = compute_degree_list(G)
    for k in expected:
        assert sorted(result[k]) == sorted(expected[k])
    assert set(result.keys()) == set(expected.keys())


def test_non_existent_neighbors():
    """
    Test a graph where a node lists a neighbor that is not present in the graph.
    The function should ignore neighbors not defined as keys in G.
    """
    G = {1: [2],
         2: [1, 3]}  # Node 3 is not present.
    # Only nodes 1 and 2 are processed.
    # For node 2: neighbors considered is only [1] so median = degree(1)=1, and since degree(2)=2 > 1, node 2 qualifies as parent.
    expected = {1: [], 2: [1]}
    result = compute_degree_list(G)
    assert set(result.keys()) == {1, 2}
    for k in expected:
        assert sorted(result[k]) == sorted(expected[k])


def test_self_loop():
    """
    Test a graph with a self-loop.
    Self-loops should not affect the parent-child assignments.
    """
    G = {1: [1, 2],  # Self-loop on 1 and connection to 2.
         2: [1]}
    # For node 1: neighbors [1,2] -> degrees: 2 (itself) and 1 for node 2. Median = (1+2)/2 = 1.5, so node 1 qualifies as parent.
    # The self-loop should be processed only once and not result in an assignment.
    expected = {1: [2], 2: []}
    result = compute_degree_list(G)
    for k in expected:
        assert sorted(result[k]) == sorted(expected[k])


def test_invalid_input():
    """
    Test that invalid input types raise appropriate errors.
    """
    # Passing None should raise an error since NoneType has no attribute 'items'.
    with pytest.raises(AttributeError):
        compute_degree_list(None)

    # Passing a list instead of a dict should raise an error.
    with pytest.raises(AttributeError):
        compute_degree_list([1, 2, 3])

    # Passing a dict with non-list neighbors should raise an error (e.g., an integer instead of an iterable).
    with pytest.raises(TypeError):
        compute_degree_list({1: 2})


def test_large_input_chain():
    """
    Test the function on a large chain graph to assess performance and correctness.
    The chain is structured so that only the second and second-last nodes become parents.
    """
    N = 10000
    G = {}
    for i in range(1, N + 1):
        if i == 1:
            G[i] = [2]
        elif i == N:
            G[i] = [N - 1]
        else:
            G[i] = [i - 1, i + 1]
    result = compute_degree_list(G)
    # For node 2: degree=2, neighbors: [1,3] with degrees [1,2] => median = 1.5 -> qualifies as parent => child 1 assigned.
    # For node N-1: degree=2, neighbors: [N-2, N] with degrees [2,1] => median = 1.5 -> qualifies as parent => child N assigned.
    assert sorted(result[2]) == [1]
    assert sorted(result[N - 1]) == [N]
    # All other nodes should have empty child lists.
    for i in range(1, N + 1):
        if i not in [2, N - 1]:
            assert result[i] == []
