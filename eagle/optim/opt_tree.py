import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class DraftNode:
    node_id: int
    parent_id: Optional[int]
    token_id: int
    depth: int
    local_prob: float
    path_logprob: float
    children: List[int] = field(default_factory=list)


@dataclass
class OptTreeConfig:
    budget: int
    overexpand_factor: float = 2.0
    score: str = "path_prob"
    ensure_connected: bool = True


def _score(node: DraftNode, score_name: str) -> float:
    if score_name in ("path_prob", "path_logprob"):
        return node.path_logprob
    raise ValueError(f"Unsupported OPT-Tree score: {score_name}")


def _topological_order(selected: Iterable[int], nodes_by_id: Dict[int, DraftNode]) -> List[int]:
    return sorted(
        selected,
        key=lambda node_id: (
            nodes_by_id[node_id].depth,
            -math.inf if nodes_by_id[node_id].parent_id is None else nodes_by_id[node_id].parent_id,
            node_id,
        ),
    )


def select_opt_tree_nodes(nodes: List[DraftNode], config: OptTreeConfig) -> List[int]:
    if config.budget < 1:
        raise ValueError("OPT-Tree budget must be at least 1")
    if not nodes:
        return []

    nodes_by_id = {node.node_id: node for node in nodes}
    roots = [node.node_id for node in nodes if node.parent_id is None]
    if len(roots) != 1:
        raise ValueError("OPT-Tree expects exactly one root node")
    root_id = roots[0]

    candidates = [node for node in nodes if node.node_id != root_id]
    candidates.sort(key=lambda node: _score(node, config.score), reverse=True)

    selected = {root_id}
    for node in candidates:
        trial = set(selected)
        cur = node.node_id
        while cur is not None:
            trial.add(cur)
            parent_id = nodes_by_id[cur].parent_id
            if parent_id is not None and parent_id not in nodes_by_id:
                raise ValueError(f"Missing parent node {parent_id} for node {cur}")
            cur = parent_id
        if len(trial) <= config.budget:
            selected = trial
        if len(selected) >= config.budget:
            break

    if config.ensure_connected:
        for node_id in selected:
            parent_id = nodes_by_id[node_id].parent_id
            if parent_id is not None and parent_id not in selected:
                raise AssertionError("Selected OPT-Tree is disconnected")

    return _topological_order(selected, nodes_by_id)
