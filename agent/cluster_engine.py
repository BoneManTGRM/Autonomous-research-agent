"""
cluster_engine.py

Cluster engine for the Autonomous Research Agent.

Goal
    Build high value clusters of related memory entries so that:
      - discoveries and hypotheses live inside coherent "idea families"
      - pruning can compress entire stale clusters instead of deleting context
      - reporting and visualization can operate on clusters instead of raw notes
      - swarm agents can target clusters for deeper repair or verification

Design
    This module is completely decoupled from tools and models.
    It does not fetch embeddings itself. You can:

      - store embeddings in memory_store entries as meta["embedding"]
      - or pass an embedding_fn callback that converts text into vectors
      - or run tag-based and recency-based clustering with no embeddings

Assumed memory_store interface:

    memory_store.list_entries() -> List[Dict[str, Any]]
        Each entry dict:
            {
              "id": str,
              "content": str,
              "meta": {
                  "rye": float (optional),
                  "tags": List[str] (optional),
                  "created_at": ISO str (optional),
                  "last_accessed": ISO str (optional),
                  "embedding": List[float] (optional),
              }
            }

    memory_store.update_entry(entry_id: str, fields: Dict[str, Any]) -> None
        Optional but powerful:
        Used to mark "cluster_root" summaries and attach cluster info.

Typical usage:

    from agent.cluster_engine import ClusterEngine

    engine = ClusterEngine(memory_store=store, run_id="run_001")

    clusters = engine.build_clusters(
        target_k=20,
        min_cluster_size=3,
        use_embeddings=True,
    )

    engine.write_cluster_roots(clusters)

You can then:
  - protect "cluster_root" tagged entries in pruning
  - use clusters for visualization and discovery summaries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Tuple
import math

# Small helpers
def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a)) or 1.0


def _cosine(a: List[float], b: List[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


@dataclass
class ClusterItem:
    entry_id: str
    text: str
    embedding: Optional[List[float]]
    tags: List[str]
    rye: Optional[float]
    created_at: Optional[datetime]
    last_accessed: Optional[datetime]


@dataclass
class ClusterStats:
    size: int
    avg_rye: Optional[float]
    max_rye: Optional[float]
    min_rye: Optional[float]
    avg_recency_days: Optional[float]
    dominant_tags: List[str]
    stability_hint: str
    discovery_potential: str


@dataclass
class Cluster:
    cluster_id: str
    items: List[ClusterItem] = field(default_factory=list)
    centroid: Optional[List[float]] = None
    stats: Optional[ClusterStats] = None
    label: str = ""


class ClusterEngine:
    """
    ClusterEngine builds and scores clusters of memory entries.

    It supports:
      - embedding-based clustering (cosine similarity)
      - tag and recency based clustering when embeddings are missing
      - per-cluster RYE and stability statistics
      - generation of cluster_root summaries to be protected by pruners
    """

    def __init__(
        self,
        memory_store: Any,
        run_id: Optional[str] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        self.memory_store = memory_store
        self.run_id = run_id
        self.embedding_fn = embedding_fn

    # --------------------------------------------
    # Public entry: build clusters from memory
    # --------------------------------------------
    def build_clusters(
        self,
        target_k: int = 20,
        min_cluster_size: int = 4,
        use_embeddings: bool = True,
        max_items: Optional[int] = None,
    ) -> List[Cluster]:
        """
        Build clusters from current memory.

        Args:
            target_k:
                Target number of clusters (soft).
            min_cluster_size:
                Clusters smaller than this are merged into nearest neighbors
                or marked as "misc".
            use_embeddings:
                If true, embedding vectors will be used when present.
                If no embeddings are available, falls back to tag and recency signal.
            max_items:
                Optional cap on number of entries to cluster for speed.

        Returns:
            List[Cluster] with stats populated.
        """
        entries = self.memory_store.list_entries()
        if max_items is not None and max_items > 0 and len(entries) > max_items:
            entries = entries[-max_items:]

        items = self._to_cluster_items(entries, use_embeddings=use_embeddings)

        if not items:
            return []

        if use_embeddings and any(it.embedding is not None for it in items):
            clusters = self._cluster_by_embeddings(items, target_k=target_k)
        else:
            clusters = self._cluster_by_tags_and_recency(items, target_k=target_k)

        clusters = self._enforce_min_cluster_size(clusters, min_cluster_size=min_cluster_size)
        clusters = self._compute_cluster_stats(clusters)
        clusters = self._label_clusters(clusters)
        return clusters

    # --------------------------------------------
    # Optional: write cluster roots into memory
    # --------------------------------------------
    def write_cluster_roots(self, clusters: List[Cluster]) -> None:
        """
        For each cluster, create or update a "cluster_root" entry in memory.

        This allows:
          - pruning engines to protect high value clusters
          - UI and reports to summarize clusters quickly
          - swarm agents to target clusters by id

        This assumes memory_store.update_entry exists. If it does not,
        this function will simply skip updating.
        """
        if not hasattr(self.memory_store, "update_entry"):
            return

        for cl in clusters:
            if not cl.items:
                continue
            stats = cl.stats or ClusterStats(
                size=len(cl.items),
                avg_rye=None,
                max_rye=None,
                min_rye=None,
                avg_recency_days=None,
                dominant_tags=[],
                stability_hint="unknown",
                discovery_potential="unknown",
            )
            root_id = f"cluster_root::{cl.cluster_id}"

            # Short preview of first few items
            preview_ids = [ci.entry_id for ci in cl.items[:5]]
            preview_tags = stats.dominant_tags[:5]

            summary_text = self._build_cluster_summary_text(cl)

            meta = {
                "tags": ["cluster_root", "equilibrium"] + stats.dominant_tags,
                "created_at": _utc_iso(),
                "last_accessed": _utc_iso(),
                "cluster_info": {
                    "cluster_id": cl.cluster_id,
                    "size": stats.size,
                    "avg_rye": stats.avg_rye,
                    "max_rye": stats.max_rye,
                    "dominant_tags": stats.dominant_tags,
                    "stability_hint": stats.stability_hint,
                    "discovery_potential": stats.discovery_potential,
                    "example_entries": preview_ids,
                    "example_tags": preview_tags,
                    "run_id": self.run_id,
                },
            }

            self.memory_store.update_entry(
                root_id,
                {
                    "content": summary_text,
                    "meta": meta,
                },
            )

    # --------------------------------------------
    # Internal: convert memory entries
    # --------------------------------------------
    def _to_cluster_items(
        self,
        entries: List[Dict[str, Any]],
        use_embeddings: bool,
    ) -> List[ClusterItem]:
        now = datetime.now(timezone.utc)
        items: List[ClusterItem] = []

        for e in entries:
            entry_id = str(e.get("id"))
            meta = e.get("meta", {}) or {}
            text = str(e.get("content", ""))

            tags = list(meta.get("tags", []) or [])
            rye = meta.get("rye")
            created_at = _parse_ts(meta.get("created_at")) or now
            last_accessed = _parse_ts(meta.get("last_accessed")) or created_at

            embedding: Optional[List[float]] = None
            if use_embeddings:
                emb = meta.get("embedding")
                if isinstance(emb, list) and emb and all(isinstance(x, (int, float)) for x in emb):
                    embedding = [float(x) for x in emb]
                elif self.embedding_fn is not None:
                    try:
                        embedding = self.embedding_fn(text)
                    except Exception:
                        embedding = None

            items.append(
                ClusterItem(
                    entry_id=entry_id,
                    text=text,
                    embedding=embedding,
                    tags=tags,
                    rye=float(rye) if rye is not None else None,
                    created_at=created_at,
                    last_accessed=last_accessed,
                )
            )

        return items

    # --------------------------------------------
    # Embedding clustering (simple k-means style)
    # --------------------------------------------
    def _cluster_by_embeddings(
        self,
        items: List[ClusterItem],
        target_k: int,
    ) -> List[Cluster]:
        vectors = [(i, i.embedding) for i in items if i.embedding is not None]
        if not vectors:
            return [Cluster(cluster_id="cluster_0", items=items)]

        # Heuristic for k
        n = len(vectors)
        k = max(2, min(target_k, int(math.sqrt(n)) + 1))

        # Initialize centroids by taking first k vectors
        centroids = [vec for (_, vec) in vectors[:k]]

        # Simple k-means loop with cosine similarity
        for _ in range(10):
            clusters: List[List[ClusterItem]] = [[] for _ in range(k)]

            for item, vec in vectors:
                best_j = 0
                best_sim = -1.0
                for j, c in enumerate(centroids):
                    sim = _cosine(vec, c)
                    if sim > best_sim:
                        best_sim = sim
                        best_j = j
                clusters[best_j].append(item)

            # Recompute centroids
            new_centroids: List[List[float]] = []
            for j in range(k):
                group = clusters[j]
                if not group:
                    new_centroids.append(centroids[j])
                    continue
                dim = len(vectors[0][1])
                acc = [0.0] * dim
                count = 0
                for item in group:
                    vec_item = item.embedding
                    if vec_item is None:
                        continue
                    for d in range(dim):
                        acc[d] += vec_item[d]
                    count += 1
                if count == 0:
                    new_centroids.append(centroids[j])
                else:
                    new_centroids.append([x / count for x in acc])

            centroids = new_centroids

        # Build Cluster objects
        clusters_out: List[Cluster] = []
        for idx in range(k):
            group = []
            for item, vec in vectors:
                # assign one more time
                best_j = 0
                best_sim = -1.0
                for j, c in enumerate(centroids):
                    sim = _cosine(vec, c)
                    if sim > best_sim:
                        best_sim = sim
                        best_j = j
                if best_j == idx:
                    group.append(item)
            if not group:
                continue
            clusters_out.append(
                Cluster(
                    cluster_id=f"emb_{idx}",
                    items=group,
                    centroid=centroids[idx],
                )
            )

        # Add items without embeddings into nearest clusters by recency
        items_no_emb = [i for i in items if i.embedding is None]
        if clusters_out and items_no_emb:
            # Fallback: attach by closest created_at to cluster average
            cluster_times: List[Tuple[Cluster, float]] = []
            for cl in clusters_out:
                if not cl.items:
                    continue
                avg_ts = sum(ci.created_at.timestamp() for ci in cl.items) / len(cl.items)
                cluster_times.append((cl, avg_ts))
            for item in items_no_emb:
                ts_val = item.created_at.timestamp()
                best_cl = min(cluster_times, key=lambda ct: abs(ct[1] - ts_val))[0]
                best_cl.items.append(item)

        return clusters_out or [Cluster(cluster_id="cluster_0", items=items)]

    # --------------------------------------------
    # Tag and recency clustering
    # --------------------------------------------
    def _cluster_by_tags_and_recency(
        self,
        items: List[ClusterItem],
        target_k: int,
    ) -> List[Cluster]:
        # Primary grouping by dominant tag
        tag_buckets: Dict[str, List[ClusterItem]] = {}
        for it in items:
            tag = it.tags[0] if it.tags else "no_tag"
            tag_buckets.setdefault(tag, []).append(it)

        clusters: List[Cluster] = []
        idx = 0
        for tag, bucket in tag_buckets.items():
            clusters.append(Cluster(cluster_id=f"tag_{tag}_{idx}", items=bucket))
            idx += 1

        # If too many tiny clusters, merge by recency
        if len(clusters) > target_k:
            # sort clusters by average recency
            def avg_ts(cl: Cluster) -> float:
                if not cl.items:
                    return 0.0
                return sum(ci.created_at.timestamp() for ci in cl.items) / len(cl.items)

            clusters.sort(key=avg_ts, reverse=True)
            # only keep top target_k clusters, merge rest into "misc"
            kept = clusters[:target_k]
            misc_items: List[ClusterItem] = []
            for cl in clusters[target_k:]:
                misc_items.extend(cl.items)
            if misc_items:
                kept.append(Cluster(cluster_id="tag_misc", items=misc_items))
            clusters = kept

        return clusters

    # --------------------------------------------
    # Enforce minimum cluster size
    # --------------------------------------------
    def _enforce_min_cluster_size(
        self,
        clusters: List[Cluster],
        min_cluster_size: int,
    ) -> List[Cluster]:
        if len(clusters) <= 1:
            return clusters

        large: List[Cluster] = []
        small: List[Cluster] = []
        for cl in clusters:
            if len(cl.items) >= min_cluster_size:
                large.append(cl)
            else:
                small.append(cl)

        if not small:
            return clusters

        if not large:
            # merge all into one
            merged_items: List[ClusterItem] = []
            for cl in small:
                merged_items.extend(cl.items)
            return [Cluster(cluster_id="merged_small", items=merged_items)]

        # merge small clusters into nearest large cluster by RYE and recency
        for cl in small:
            if not cl.items:
                continue
            avg_rye = self._cluster_avg_rye(cl)
            avg_ts = self._cluster_avg_ts(cl)
            best_idx = 0
            best_score = -1e9
            for i, big in enumerate(large):
                big_avg_rye = self._cluster_avg_rye(big)
                big_avg_ts = self._cluster_avg_ts(big)
                score = 0.0
                if avg_rye is not None and big_avg_rye is not None:
                    score += -abs(avg_rye - big_avg_rye)
                if avg_ts is not None and big_avg_ts is not None:
                    score += -abs(avg_ts - big_avg_ts) / 86400.0
                if score > best_score:
                    best_score = score
                    best_idx = i
            large[best_idx].items.extend(cl.items)

        return large

    # --------------------------------------------
    # Cluster stats
    # --------------------------------------------
    def _cluster_avg_rye(self, cl: Cluster) -> Optional[float]:
        vals = [ci.rye for ci in cl.items if ci.rye is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _cluster_avg_ts(self, cl: Cluster) -> Optional[float]:
        vals = [ci.created_at.timestamp() for ci in cl.items if ci.created_at is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _compute_cluster_stats(self, clusters: List[Cluster]) -> List[Cluster]:
        now = datetime.now(timezone.utc)
        for cl in clusters:
            size = len(cl.items)
            rye_vals = [ci.rye for ci in cl.items if ci.rye is not None]
            avg_rye = sum(rye_vals) / len(rye_vals) if rye_vals else None
            max_rye = max(rye_vals) if rye_vals else None
            min_rye = min(rye_vals) if rye_vals else None

            # recency
            ages: List[float] = []
            for ci in cl.items:
                ref = ci.last_accessed or ci.created_at or now
                ages.append((now - ref).total_seconds() / 86400.0)
            avg_recency_days = sum(ages) / len(ages) if ages else None

            # dominant tags
            tag_counts: Dict[str, int] = {}
            for ci in cl.items:
                for t in ci.tags:
                    tag_counts[t] = tag_counts.get(t, 0) + 1
            dominant_tags = sorted(tag_counts, key=lambda t: tag_counts[t], reverse=True)[:5]

            # stability and discovery hints
            stability_hint = self._infer_stability_hint(avg_rye, avg_recency_days)
            discovery_potential = self._infer_discovery_potential(avg_rye, size, dominant_tags)

            cl.stats = ClusterStats(
                size=size,
                avg_rye=avg_rye,
                max_rye=max_rye,
                min_rye=min_rye,
                avg_recency_days=avg_recency_days,
                dominant_tags=dominant_tags,
                stability_hint=stability_hint,
                discovery_potential=discovery_potential,
            )
        return clusters

    def _infer_stability_hint(
        self,
        avg_rye: Optional[float],
        avg_recency_days: Optional[float],
    ) -> str:
        if avg_rye is None:
            return "unknown"
        if avg_rye >= 0.20:
            return "high_efficiency"
        if avg_rye >= 0.10:
            return "stable_equilibrium"
        if avg_rye < 0.02:
            return "low_yield_or_collapse_zone"
        return "transient_or_mixed"

    def _infer_discovery_potential(
        self,
        avg_rye: Optional[float],
        size: int,
        tags: List[str],
    ) -> str:
        if avg_rye is None:
            return "unknown"
        if any(t in ("discovery", "hypothesis", "verified") for t in tags) and avg_rye >= 0.15:
            return "very_high"
        if avg_rye >= 0.15 and size >= 8:
            return "high"
        if avg_rye >= 0.08:
            return "moderate"
        return "low"

    # --------------------------------------------
    # Cluster labeling and summary text
    # --------------------------------------------
    def _label_clusters(self, clusters: List[Cluster]) -> List[Cluster]:
        for idx, cl in enumerate(clusters):
            st = cl.stats
            if st is None:
                cl.label = f"Cluster {idx}"
                continue
            tag_part = ", ".join(st.dominant_tags[:3]) if st.dominant_tags else "no tags"
            rye_str = f"{st.avg_rye:.3f}" if st.avg_rye is not None else "n/a"
            cl.label = f"[{idx}] tags: {tag_part} | avg RYE: {rye_str} | size: {st.size}"
        return clusters

    def _build_cluster_summary_text(self, cl: Cluster) -> str:
        st = cl.stats
        lines: List[str] = []
        lines.append(f"# Cluster {cl.cluster_id}")
        if st:
            lines.append("")
            lines.append(f"- size: {st.size}")
            lines.append(f"- avg RYE: {st.avg_rye:.3f}" if st.avg_rye is not None else "- avg RYE: n/a")
            lines.append(f"- max RYE: {st.max_rye:.3f}" if st.max_rye is not None else "- max RYE: n/a")
            lines.append(f"- dominant tags: {', '.join(st.dominant_tags) if st.dominant_tags else 'none'}")
            lines.append(f"- stability hint: {st.stability_hint}")
            lines.append(f"- discovery potential: {st.discovery_potential}")
        lines.append("")
        lines.append("## Example notes")
        for ci in cl.items[:5]:
            preview = ci.text.strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            lines.append(f"- [{ci.entry_id}] {preview}")
        return "\n".join(lines)
