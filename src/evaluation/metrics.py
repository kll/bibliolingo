"""Evaluation metrics for BiblioLingo retrieval."""

import logging
from typing import List, Dict
import numpy as np

logger = logging.getLogger(__name__)


class RetrievalMetrics:
    """
    Calculate retrieval metrics: Recall@k, MRR, Precision@k.

    Metrics are calculated based on whether retrieved documents contain
    any of the target document IDs.
    """

    def recall_at_k(
        self,
        retrieved_doc_ids: List[str],
        target_doc_ids: List[str],
        k: int,
    ) -> float:
        """
        Calculate Recall@k: Did we retrieve at least one relevant document in top-k?

        Args:
            retrieved_doc_ids: List of retrieved document IDs (ordered by rank)
            target_doc_ids: List of relevant document IDs
            k: Number of top results to consider

        Returns:
            1.0 if at least one relevant doc in top-k, 0.0 otherwise
        """
        if not target_doc_ids:
            return 0.0

        # Get top-k retrieved docs
        top_k = retrieved_doc_ids[:k]

        # Check if any target doc is in top-k
        for target_id in target_doc_ids:
            if self._doc_id_matches(target_id, top_k):
                return 1.0

        return 0.0

    def mean_reciprocal_rank(
        self,
        retrieved_doc_ids: List[str],
        target_doc_ids: List[str],
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR): 1 / rank of first relevant document.

        Args:
            retrieved_doc_ids: List of retrieved document IDs (ordered by rank)
            target_doc_ids: List of relevant document IDs

        Returns:
            1 / rank of first relevant doc (0.0 if no relevant docs found)
        """
        if not target_doc_ids:
            return 0.0

        # Find rank of first relevant document
        for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
            for target_id in target_doc_ids:
                if self._doc_id_matches_single(target_id, doc_id):
                    return 1.0 / rank

        return 0.0

    def precision_at_k(
        self,
        retrieved_doc_ids: List[str],
        target_doc_ids: List[str],
        k: int,
    ) -> float:
        """
        Calculate Precision@k: Fraction of top-k that are relevant.

        Args:
            retrieved_doc_ids: List of retrieved document IDs (ordered by rank)
            target_doc_ids: List of relevant document IDs
            k: Number of top results to consider

        Returns:
            Precision in range [0.0, 1.0]
        """
        if not target_doc_ids or k == 0:
            return 0.0

        # Get top-k retrieved docs
        top_k = retrieved_doc_ids[:k]

        # Count relevant docs in top-k
        relevant_count = 0
        for doc_id in top_k:
            for target_id in target_doc_ids:
                if self._doc_id_matches_single(target_id, doc_id):
                    relevant_count += 1
                    break  # Count each doc only once

        return relevant_count / min(k, len(top_k))

    def _doc_id_matches(self, target_id: str, retrieved_ids: List[str]) -> bool:
        """
        Check if target_id matches any of the retrieved_ids.

        Handles partial matches (e.g., "confluence-3074097170" matches chunk IDs
        that start with "confluence-3074097170-").

        Args:
            target_id: Target document ID (possibly without chunk suffix)
            retrieved_ids: List of retrieved document IDs (may include chunk IDs)

        Returns:
            True if target matches any retrieved ID
        """
        for retrieved_id in retrieved_ids:
            if self._doc_id_matches_single(target_id, retrieved_id):
                return True
        return False

    def _doc_id_matches_single(self, target_id: str, retrieved_id: str) -> bool:
        """
        Check if target_id matches retrieved_id.

        Handles:
        - Exact match
        - Chunk ID match (retrieved_id starts with target_id)

        Args:
            target_id: Target document ID
            retrieved_id: Retrieved document ID (possibly chunk ID)

        Returns:
            True if IDs match
        """
        # Exact match
        if target_id == retrieved_id:
            return True

        # Check if retrieved_id is a chunk of target_id
        # E.g., target="confluence-3074097170", retrieved="confluence-3074097170-decision-0"
        if retrieved_id.startswith(target_id + "-"):
            return True

        # Also check doc_id field if it's embedded in retrieved_id
        # This handles cases where we store doc_id separately
        return False

    def calculate_aggregate_metrics(
        self, query_results: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate aggregate metrics across all queries.

        Args:
            query_results: List of per-query results with metrics

        Returns:
            Dictionary of aggregate metrics
        """
        if not query_results:
            return {}

        # Collect metrics by type
        metrics_by_k = {}

        for result in query_results:
            for metric_name, value in result.items():
                if metric_name not in ["query", "target_docs", "retrieved_docs"]:
                    if metric_name not in metrics_by_k:
                        metrics_by_k[metric_name] = []
                    metrics_by_k[metric_name].append(value)

        # Calculate means
        aggregate = {}
        for metric_name, values in metrics_by_k.items():
            aggregate[metric_name] = float(np.mean(values))

        return aggregate
