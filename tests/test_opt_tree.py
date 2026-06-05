from eagle.optim.opt_tree import DraftNode, OptTreeConfig, select_opt_tree_nodes


def make_tree():
    return [
        DraftNode(0, None, -1, 0, 1.0, 0.0),
        DraftNode(1, 0, 10, 1, 0.9, -0.1),
        DraftNode(2, 0, 20, 1, 0.4, -0.9),
        DraftNode(3, 1, 11, 2, 0.8, -0.3),
        DraftNode(4, 1, 12, 2, 0.2, -1.7),
        DraftNode(5, 2, 21, 2, 0.95, -0.95),
        DraftNode(6, 3, 13, 3, 0.7, -0.65),
    ]


def test_path_logprob_monotonic():
    nodes = {node.node_id: node for node in make_tree()}
    for node in nodes.values():
        if node.parent_id is not None:
            assert node.path_logprob <= nodes[node.parent_id].path_logprob


def test_selected_tree_connected():
    nodes = make_tree()
    selected = set(select_opt_tree_nodes(nodes, OptTreeConfig(budget=4)))
    by_id = {node.node_id: node for node in nodes}
    for node_id in selected:
        parent_id = by_id[node_id].parent_id
        if parent_id is not None:
            assert parent_id in selected


def test_budget_not_exceeded():
    selected = select_opt_tree_nodes(make_tree(), OptTreeConfig(budget=3))
    assert len(selected) <= 3


def test_high_prob_path_preferred():
    selected = set(select_opt_tree_nodes(make_tree(), OptTreeConfig(budget=4)))
    assert {0, 1, 3}.issubset(selected)
    assert 4 not in selected
