from __future__ import annotations

from ..models import KnowledgeGraph, KnowledgeViewMode


def knowledge_graph_to_mermaid(graph: KnowledgeGraph) -> str:
    """Render a deterministic view; repositories and relations remain authoritative."""

    aliases = {
        node.node_id: f"n_{node.node_id.removeprefix('knowledge-node-')}"
        for node in graph.nodes
    }
    lines = ["flowchart LR"]
    for node in graph.nodes:
        label = f"{node.record_type}: {node.record_id}"
        if graph.view_mode == KnowledgeViewMode.AUDIT:
            label += f"\\nstatus={node.curation_status or 'not_curated'}"
        label = _escape(label)
        lines.append(f'  {aliases[node.node_id]}["{label}"]')
    for edge in graph.edges:
        label = str(edge.predicate)
        if graph.view_mode == KnowledgeViewMode.AUDIT:
            label += f"\\nstatus={edge.curation_status or 'not_curated'}"
        lines.append(
            f"  {aliases[edge.subject_node_id]} -->|{label}| "
            f"{aliases[edge.object_node_id]}"
        )
    return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
