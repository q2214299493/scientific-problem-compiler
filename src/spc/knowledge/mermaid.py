from __future__ import annotations

from ..models import KnowledgeGraph


def knowledge_graph_to_mermaid(graph: KnowledgeGraph) -> str:
    """Render a deterministic view; repositories and relations remain authoritative."""

    aliases = {
        node.node_id: f"n_{node.node_id.removeprefix('knowledge-node-')}"
        for node in graph.nodes
    }
    lines = ["flowchart LR"]
    for node in graph.nodes:
        label = _escape(f"{node.record_type}: {node.record_id}")
        lines.append(f'  {aliases[node.node_id]}["{label}"]')
    for edge in graph.edges:
        lines.append(
            f"  {aliases[edge.subject_node_id]} -->|{edge.predicate}| "
            f"{aliases[edge.object_node_id]}"
        )
    return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
