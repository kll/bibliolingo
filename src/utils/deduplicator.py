"""MinHash-based deduplication with priority selection for BiblioLingo."""

import logging
from typing import List, Dict, Set
from collections import defaultdict
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Deduplicates chunks using MinHash for near-duplicate detection.

    Applies priority-based selection: ADR > GitHub > Confluence DEV
    """

    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        """
        Initialize deduplicator.

        Args:
            threshold: Jaccard similarity threshold (0-1) for duplicates
            num_perm: Number of permutations for MinHash
        """
        self.threshold = threshold
        self.num_perm = num_perm

    def deduplicate(self, chunks: List) -> tuple[List, Dict[str, str]]:
        """
        Deduplicate chunks using MinHash.

        Args:
            chunks: List of Chunk objects

        Returns:
            Tuple of (deduplicated_chunks, duplicate_mapping)
            duplicate_mapping: Maps duplicate chunk_id -> canonical chunk_id
        """
        if not chunks:
            return [], {}

        logger.info(f"Deduplicating {len(chunks)} chunks (threshold={self.threshold})")

        # Create MinHash LSH index
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        # Generate MinHash signatures
        chunk_minhashes = {}
        for chunk in chunks:
            minhash = self._create_minhash(chunk.content)
            chunk_minhashes[chunk.chunk_id] = minhash

        # Find duplicate clusters
        duplicate_clusters = defaultdict(list)
        processed = set()

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.chunk_id

            if chunk_id in processed:
                continue

            # Query LSH for similar chunks
            minhash = chunk_minhashes[chunk_id]

            # Get all chunks that haven't been indexed yet
            remaining_chunks = [c for c in chunks[i:] if c.chunk_id not in processed]

            # Find duplicates by comparing with remaining chunks
            cluster = [chunk]
            for other_chunk in remaining_chunks:
                if other_chunk.chunk_id == chunk_id:
                    continue

                other_minhash = chunk_minhashes[other_chunk.chunk_id]
                similarity = minhash.jaccard(other_minhash)

                if similarity >= self.threshold:
                    cluster.append(other_chunk)
                    processed.add(other_chunk.chunk_id)

            if len(cluster) > 1:
                # Found duplicates
                duplicate_clusters[chunk_id] = cluster
                logger.debug(
                    f"Found duplicate cluster of {len(cluster)} chunks for {chunk_id}"
                )

            processed.add(chunk_id)

        # Select canonical chunks based on priority
        canonical_chunks = []
        duplicate_mapping = {}

        # Add non-duplicate chunks
        for chunk in chunks:
            if chunk.chunk_id not in duplicate_clusters:
                canonical_chunks.append(chunk)

        # Process duplicate clusters
        for cluster_id, cluster in duplicate_clusters.items():
            # Select chunk with highest priority
            canonical = self._select_canonical(cluster)
            canonical_chunks.append(canonical)

            # Map all non-canonical chunks to canonical
            for chunk in cluster:
                if chunk.chunk_id != canonical.chunk_id:
                    duplicate_mapping[chunk.chunk_id] = canonical.chunk_id

        logger.info(
            f"Deduplication complete: {len(chunks)} -> {len(canonical_chunks)} chunks "
            f"({len(duplicate_mapping)} duplicates removed)"
        )

        return canonical_chunks, duplicate_mapping

    def _create_minhash(self, text: str) -> MinHash:
        """
        Create MinHash signature for text.

        Args:
            text: Text content

        Returns:
            MinHash object
        """
        minhash = MinHash(num_perm=self.num_perm)

        # Tokenize and add to MinHash
        # Use word-level tokens for better similarity detection
        tokens = text.lower().split()
        for token in tokens:
            minhash.update(token.encode('utf8'))

        return minhash

    def _select_canonical(self, cluster: List) -> object:
        """
        Select canonical chunk from duplicate cluster based on priority.

        Priority order:
        1. Highest priority_score
        2. Most recent updated_at
        3. Longest content

        Args:
            cluster: List of duplicate Chunk objects

        Returns:
            Canonical Chunk object
        """
        # Sort by priority (desc), updated_at (desc), content length (desc)
        sorted_cluster = sorted(
            cluster,
            key=lambda c: (
                c.priority_score,
                c.updated_at or "",
                len(c.content),
            ),
            reverse=True,
        )

        canonical = sorted_cluster[0]

        logger.debug(
            f"Selected canonical chunk: {canonical.chunk_id} "
            f"(priority={canonical.priority_score}, "
            f"source={canonical.source}, "
            f"doc_type={canonical.doc_type})"
        )

        return canonical

    def get_deduplication_stats(
        self, original_count: int, deduplicated_count: int, duplicate_mapping: Dict
    ) -> Dict:
        """
        Generate deduplication statistics.

        Args:
            original_count: Original number of chunks
            deduplicated_count: Number after deduplication
            duplicate_mapping: Duplicate chunk mapping

        Returns:
            Statistics dictionary
        """
        removed_count = original_count - deduplicated_count
        removal_rate = removed_count / original_count if original_count > 0 else 0

        stats = {
            "original_chunks": original_count,
            "deduplicated_chunks": deduplicated_count,
            "duplicates_removed": removed_count,
            "removal_rate": round(removal_rate, 3),
            "duplicate_clusters": len(set(duplicate_mapping.values())),
        }

        return stats
