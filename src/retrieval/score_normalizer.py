"""Score normalizer for BiblioLingo hybrid retrieval results."""

import logging
from typing import List
from langchain_core.documents import Document

from src.utils.config import config

logger = logging.getLogger(__name__)


class ScoreNormalizer:
    """
    Normalizes document scores from hybrid retrieval.

    Provides score normalization (0-1 range) and confidence checking to assess
    result quality. Results are sorted by RRF relevance scores from BM25 and
    vector search without any additional boosting or reranking.
    """

    def __init__(
        self,
        confidence_threshold: float = None,
        rrf_k: int = None,
        alpha: float = None,
    ):
        """
        Initialize score normalizer.

        Args:
            confidence_threshold: Score threshold for triggering fallback (0-1 scale)
            rrf_k: RRF constant from hybrid retriever for score normalization
            alpha: BM25 weight from hybrid retriever for score normalization
        """
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.confidence_threshold
        )
        self.rrf_k = rrf_k if rrf_k is not None else config.rrf_k
        self.alpha = alpha if alpha is not None else config.default_alpha

        # Calculate theoretical maximum score for normalization
        self.max_possible_score = self._calculate_max_possible_score()

    def _calculate_max_possible_score(self) -> float:
        """
        Calculate the theoretical maximum possible score.

        This is the score a document would get if it ranks #1 (rank=0) in both
        BM25 and vector search. No source boosting is applied.

        Returns:
            Maximum possible RRF score
        """
        # Maximum RRF score (rank 0 in both lists)
        # After weight normalization: weight_bm25 = alpha, weight_vector = (1-alpha)
        # Total weights sum to 1.0
        # RRF formula: weight / (k + rank + 1)
        max_rrf_score = self.alpha / (self.rrf_k + 1) + (1.0 - self.alpha) / (
            self.rrf_k + 1
        )

        logger.debug(
            f"Calculated max possible score: {max_rrf_score:.6f} "
            f"(rrf_k={self.rrf_k}, alpha={self.alpha})"
        )

        return max_rrf_score

    def rerank(self, documents: List[Document]) -> List[Document]:
        """
        Normalize document scores to 0-1 range.

        Does not apply any boosting - documents are sorted purely by their
        RRF scores from hybrid retrieval (BM25 + vector).

        Args:
            documents: List of documents with rrf_score in metadata

        Returns:
            Documents sorted by RRF score with normalized scores
        """
        if not documents:
            return documents

        logger.debug(f"Normalizing scores for {len(documents)} documents")

        # Normalize scores without boosting
        for doc in documents:
            rrf_score = doc.metadata.get("rrf_score", 0.0)
            source = doc.metadata.get("source", "unknown")

            # Normalize to 0-1 range (no boosting)
            normalized_score = rrf_score / self.max_possible_score

            # Store scores
            doc.metadata["normalized_score"] = normalized_score
            doc.metadata["final_score"] = normalized_score

            logger.debug(
                f"  {doc.metadata.get('chunk_id', 'unknown')} ({source}): "
                f"rrf={rrf_score:.4f}, normalized={normalized_score:.4f}"
            )

        # Sort by RRF score (descending)
        reranked = sorted(
            documents,
            key=lambda d: d.metadata.get("rrf_score", 0.0),
            reverse=True,
        )

        return reranked

    def check_confidence(self, documents: List[Document]) -> tuple[bool, float]:
        """
        Check if top result meets confidence threshold.

        Uses normalized scores (0-1 range) where 1.0 is the theoretical maximum
        (rank #1 in both BM25 and vector search).

        Args:
            documents: Documents with normalized scores

        Returns:
            Tuple of (is_confident, top_score) where top_score is normalized (0-1)
        """
        if not documents:
            return False, 0.0

        top_score = documents[0].metadata.get("final_score", 0.0)
        is_confident = top_score >= self.confidence_threshold

        if not is_confident:
            logger.warning(
                f"Low confidence: top score {top_score:.4f} < "
                f"threshold {self.confidence_threshold:.4f} "
                f"(normalized 0-1 scale)"
            )
        else:
            logger.debug(
                f"Confidence OK: top score {top_score:.4f} >= "
                f"threshold {self.confidence_threshold:.4f}"
            )

        return is_confident, top_score

    def suggest_fallback_filters(
        self, original_filters: dict, confidence_level: float
    ) -> dict:
        """
        Suggest relaxed filters for fallback search.

        Args:
            original_filters: Original MongoDB filters
            confidence_level: Confidence score of top result

        Returns:
            Relaxed filters
        """
        if confidence_level < 0.2:
            # Very low confidence: remove all filters
            logger.info("Very low confidence, removing all filters")
            return {}

        if confidence_level < self.confidence_threshold:
            # Low confidence: remove doc_type filter but keep source
            relaxed = original_filters.copy()
            if "doc_type" in relaxed:
                logger.info("Low confidence, removing doc_type filter")
                del relaxed["doc_type"]
            return relaxed

        # Confidence is acceptable
        return original_filters
