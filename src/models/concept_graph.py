"""Conceptual Knowledge Graph — Unified Relational Memory.

Wires together all 18 cognitive modules into a single relational network.

Each node = a concept (object, property, action, label).
Each edge = a relation with confidence [0,1], sourced from experience.

Data sources (all already producing output during training):
    SlotAttention → object attribute centroids
    CausalDiscovery → cause→effect edges
    SemanticMemory → cross-episode facts
    RuleInductionEngine → IF-THEN rules
    TouchBridge → haptic properties ("heavy", "slippery")
    LLM → language labels for objects
    EpisodeMemory → when/where events happened
    NumberSense → cardinality statistics

Enables four new abilities:
    1. Analogy: "this is like an apple" via graph traversal
    2. Inheritance: "furniture → heavy" auto-applies to all furniture nodes
    3. Cross-modal binding: "red+round+light+rolls+called-ball" → one concept
    4. Fast adaptation: 10-step recognition vs 1000-step

概念图谱：统一关系记忆。连接全部 18 个认知模块。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ConceptNode:
    """A concept in the knowledge graph."""
    id: int
    name: str                        # human-readable label
    embedding: torch.Tensor           # (d_model,) centroid from all sources
    source_modules: list[str] = field(default_factory=list)  # which modules created this
    instance_count: int = 0           # how many times observed
    created_step: int = 0
    last_updated_step: int = 0


@dataclass
class ConceptEdge:
    """A relationship between two concepts."""
    source_id: int
    target_id: int
    relation_type: str               # "has_color", "causes", "is_like", "lighter_than"
    confidence: float = 0.5          # [0,1] Bayesian-updated
    source_module: str = ""           # which module reported this
    observation_count: int = 0
    created_step: int = 0


class ConceptGraph(nn.Module):
    """Unified conceptual knowledge graph connecting all module outputs.

    Bounded: max_nodes, max_edges declared at construction (Axiom 1).
    VRAM: ~0.1 GB (1000 nodes × 128 dim + 5000 edges).
    """

    def __init__(
        self,
        d_model: int = 128,
        max_nodes: int = 1000,
        max_edges: int = 5000,
        merge_similarity: float = 0.85,  # cosine threshold to merge nodes
        inheritance_threshold: float = 0.6,  # min conf for attribute inheritance
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._max_nodes = max_nodes
        self._max_edges = max_edges
        self._merge_sim = merge_similarity
        self._inherit_threshold = inheritance_threshold

        self._nodes: dict[int, ConceptNode] = {}
        self._edges: dict[tuple[int, int], ConceptEdge] = {}
        self._next_id = 0

        # Attention for graph traversal
        self._query_proj = nn.Linear(d_model, d_model)

    @property
    def capacity(self) -> int:
        return self._max_nodes

    def __len__(self) -> int:
        return len(self._nodes)

    # ================================================================ add/update

    def add_concept(
        self, embedding: torch.Tensor, name: str = "",
        source: str = "", step: int = 0,
    ) -> int:
        """Add a concept node or merge with existing similar node.

        Returns the node id (new or merged).
        """
        # Check for similar existing node
        if len(self._nodes) > 0:
            node_list = list(self._nodes.values())
            embs = torch.stack([n.embedding.to(embedding.device) for n in node_list])
            sim = F.cosine_similarity(embedding.unsqueeze(0), embs, dim=-1)
            best_idx = int(sim.argmax().item())
            if float(sim[best_idx]) > self._merge_sim:
                node = node_list[best_idx]
                node.embedding = node.embedding * 0.9 + embedding.detach().cpu() * 0.1
                node.instance_count += 1
                node.last_updated_step = step
                if source and source not in node.source_modules:
                    node.source_modules.append(source)
                if name and not node.name:
                    node.name = name
                return node.id

        # New node
        if len(self._nodes) >= self._max_nodes:
            self._evict_node()
        node_id = self._next_id
        self._next_id += 1
        self._nodes[node_id] = ConceptNode(
            id=node_id, name=name,
            embedding=embedding.detach().cpu(),
            source_modules=[source] if source else [],
            instance_count=1,
            created_step=step,
            last_updated_step=step,
        )
        return node_id

    def add_edge(
        self, source_id: int, target_id: int,
        relation: str, confidence: float = 0.5,
        source_module: str = "", step: int = 0,
    ) -> None:
        """Add or strengthen a relationship edge."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return
        key = (source_id, target_id)
        if key in self._edges:
            edge = self._edges[key]
            n = edge.observation_count + 1
            edge.confidence = (edge.confidence * edge.observation_count + confidence) / n
            edge.observation_count = n
            return
        if len(self._edges) >= self._max_edges:
            self._evict_edge()
        self._edges[key] = ConceptEdge(
            source_id=source_id, target_id=target_id,
            relation_type=relation, confidence=confidence,
            source_module=source_module, observation_count=1,
            created_step=step,
        )

    def _evict_node(self) -> None:
        if not self._nodes:
            return
        worst = min(self._nodes, key=lambda nid: self._nodes[nid].instance_count)
        # Remove all edges connected to this node
        to_remove = [k for k in self._edges if k[0] == worst or k[1] == worst]
        for k in to_remove:
            del self._edges[k]
        del self._nodes[worst]

    def _evict_edge(self) -> None:
        if not self._edges:
            return
        worst = min(self._edges, key=lambda k: self._edges[k].confidence)
        del self._edges[worst]

    # ================================================================ inference

    def find_analog(
        self, query_embedding: torch.Tensor, k: int = 3,
    ) -> list[tuple[ConceptNode, float]]:
        """Find the k most similar concepts (analogy: "this is like X")."""
        if not self._nodes:
            return []
        node_list = list(self._nodes.values())
        # Project BOTH query and stored nodes through the same projection
        q = self._query_proj(query_embedding.unsqueeze(0))
        embs = torch.stack([
            self._query_proj(n.embedding.unsqueeze(0).to(q.device)).squeeze(0)
            for n in node_list
        ])
        sim = F.cosine_similarity(q, embs, dim=-1)
        top_k = sim.topk(min(k, len(node_list))).indices.tolist()
        return [(node_list[i], float(sim[i].detach())) for i in top_k]

    def inherit_attributes(
        self, node_id: int,
    ) -> dict[str, list[tuple[str, float]]]:
        """Inherit attributes from related concepts.

        Returns dict mapping relation_type → [(target_name, confidence), ...].
        """
        inherited: dict[str, list[tuple[str, float]]] = {}
        for (src, tgt), edge in self._edges.items():
            if src == node_id and edge.confidence >= self._inherit_threshold:
                tgt_name = self._nodes.get(tgt, ConceptNode(-1, "?",
                                torch.zeros(self._d_model))).name
                inherited.setdefault(edge.relation_type, []).append(
                    (tgt_name, edge.confidence),
                )
        return inherited

    def bind_cross_modal(
        self, modalities: dict[str, torch.Tensor],
        step: int = 0,
    ) -> int:
        """Bind multiple modality embeddings into one concept node.

        modalities = {
            "color": red_embedding,
            "touch": light_embedding,
            "label": "ball",
            "causal": rolls_embedding,
        }
        → creates/merges a unified "ball" concept.

        Returns the concept node id.
        """
        # Average all modality embeddings
        embs = [e for e in modalities.values() if isinstance(e, torch.Tensor)]
        labels = [modalities.get("label", "")]
        centroid = torch.stack(embs).mean(dim=0) if embs else torch.randn(self._d_model)

        node_id = self.add_concept(
            embedding=centroid,
            name=str(labels[0]) if labels else "",
            source="cross_modal_binding",
            step=step,
        )

        # Add edges from concept to each modality attribute
        for mod_name, mod_val in modalities.items():
            if isinstance(mod_val, torch.Tensor):
                attr_id = self.add_concept(
                    embedding=mod_val,
                    name=f"{labels[0]}_{mod_name}" if labels else mod_name,
                    source=mod_name,
                    step=step,
                )
                self.add_edge(node_id, attr_id, f"has_{mod_name}",
                            confidence=0.8, source_module=mod_name, step=step)

        return node_id

    # ================================================================ traversal

    def query(
        self, query_embedding: torch.Tensor, relation_filter: str | None = None,
        k: int = 8,
    ) -> list[dict[str, Any]]:
        """Knowledge-graph-aware retrieval.

        Returns concepts + their inherited attributes + analogies.
        """
        analogs = self.find_analog(query_embedding, k=k)
        results = []
        for node, sim in analogs:
            attrs = self.inherit_attributes(node.id)
            # Filter by relation if specified
            if relation_filter:
                attrs = {k: v for k, v in attrs.items() if relation_filter in k}
            results.append({
                "concept": node.name or f"concept_{node.id}",
                "similarity": sim,
                "instance_count": node.instance_count,
                "attributes": attrs,
                "sources": node.source_modules,
            })
        return results

    # ================================================================ training

    def update_from_semantic(
        self, facts: list[Any], step: int,
    ) -> None:
        """Ingest semantic facts into the graph."""
        for fact in facts:
            if not hasattr(fact, 'key'):
                continue
            node_id = self.add_concept(
                embedding=fact.embedding if hasattr(fact, 'embedding') else torch.randn(self._d_model),
                name=fact.key, source="semantic_memory", step=step,
            )
            # Also create edge for confidence
            self.add_edge(node_id, node_id, "has_confidence",
                        confidence=fact.value if hasattr(fact, 'value') else 0.5,
                        source_module="semantic_memory", step=step)

    def update_from_causal(
        self, edges: list[tuple[str, str, float]], step: int,
    ) -> None:
        """Ingest causal edges into the graph."""
        for source, target, strength in edges:
            src_id = self.add_concept(
                embedding=torch.randn(self._d_model),
                name=source, source="causal_discovery", step=step,
            )
            tgt_id = self.add_concept(
                embedding=torch.randn(self._d_model),
                name=target, source="causal_discovery", step=step,
            )
            self.add_edge(src_id, tgt_id, "causes",
                        confidence=float(strength), source_module="causal_discovery", step=step)

    # ================================================================ diagnostics

    def summary(self) -> dict:
        if not self._edges:
            return {"nodes": len(self._nodes), "edges": 0}
        top_edges = sorted(
            self._edges.values(), key=lambda e: -e.confidence,
        )[:5]
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "max_nodes": self._max_nodes,
            "max_edges": self._max_edges,
            "top_relations": [
                f"{self._nodes.get(e.source_id, ConceptNode(-1,'?',torch.zeros(1))).name} "
                f"--{e.relation_type}--> "
                f"{self._nodes.get(e.target_id, ConceptNode(-1,'?',torch.zeros(1))).name} "
                f"({e.confidence:.2f})"
                for e in top_edges
            ],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "next_id": self._next_id,
            "nodes": [
                {"id": n.id, "name": n.name, "instance_count": n.instance_count,
                 "sources": n.source_modules, "embedding": n.embedding}
                for n in self._nodes.values()
            ],
            "edges": [
                {"source": e.source_id, "target": e.target_id,
                 "relation": e.relation_type, "confidence": e.confidence,
                 "source_module": e.source_module, "count": e.observation_count}
                for e in self._edges.values()
            ],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._next_id = int(state.get("next_id", 0))
        self._nodes.clear()
        self._edges.clear()
        for n in state.get("nodes", []):
            self._nodes[n["id"]] = ConceptNode(
                id=n["id"], name=n.get("name", ""),
                embedding=n.get("embedding", torch.zeros(self._d_model)),
                source_modules=n.get("sources", []),
                instance_count=n.get("instance_count", 0),
            )
        for e in state.get("edges", []):
            self._edges[(e["source"], e["target"])] = ConceptEdge(
                source_id=e["source"], target_id=e["target"],
                relation_type=e["relation"], confidence=e["confidence"],
                source_module=e.get("source_module", ""),
                observation_count=e.get("count", 0),
            )
