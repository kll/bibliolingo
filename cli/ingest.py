"""Ingestion CLI for BiblioLingo."""

import logging
import click
from pathlib import Path

from src.ingestion.loader import MarkdownLoader
from src.ingestion.normalizer import MetadataNormalizer
from src.ingestion.chunker import MarkdownChunker
from src.utils.deduplicator import Deduplicator
from src.ingestion.indexer import Indexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data-dir",
    default="./data",
    help="Directory containing markdown files",
    type=click.Path(exists=True),
)
@click.option(
    "--artifacts-dir",
    default="./artifacts",
    help="Directory for caching artifacts",
    type=click.Path(),
)
@click.option(
    "--batch-size",
    default=100,
    help="Batch size for embedding generation",
    type=int,
)
@click.option(
    "--dedupe-threshold",
    default=0.85,
    help="MinHash similarity threshold for deduplication (0-1)",
    type=float,
)
def ingest(data_dir: str, artifacts_dir: str, batch_size: int, dedupe_threshold: float):
    """
    Ingest markdown documents into BiblioLingo.

    This command:
    1. Loads markdown files from data directory
    2. Extracts metadata and normalizes
    3. Chunks documents by headings (ADR-aware)
    4. Deduplicates using MinHash
    5. Generates embeddings via OpenAI
    6. Stores in MongoDB with vector index
    """
    logger.info("=" * 80)
    logger.info("BiblioLingo Ingestion Pipeline")
    logger.info("=" * 80)

    # Step 1: Load documents
    logger.info("\n[1/5] Loading markdown documents...")
    loader = MarkdownLoader(data_dir=data_dir)
    documents = loader.load_all()

    if not documents:
        logger.error("No documents found. Exiting.")
        return

    logger.info(f"✓ Loaded {len(documents)} documents")

    # Step 2: Extract metadata
    logger.info("\n[2/5] Extracting and normalizing metadata...")
    normalizer = MetadataNormalizer()

    doc_metadata = []
    for doc in documents:
        metadata = normalizer.normalize(doc)
        doc_metadata.append((doc, metadata))

    logger.info(f"✓ Normalized metadata for {len(doc_metadata)} documents")

    # Log doc type distribution
    doc_type_counts = {}
    for _, metadata in doc_metadata:
        doc_type = metadata.doc_type
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

    logger.info("  Doc type distribution:")
    for doc_type, count in sorted(doc_type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"    {doc_type}: {count}")

    # Step 3: Chunk documents
    logger.info("\n[3/5] Chunking documents...")
    chunker = MarkdownChunker()

    all_chunks = []
    for doc, metadata in doc_metadata:
        chunks = chunker.chunk_document(doc, metadata)
        all_chunks.extend(chunks)

    logger.info(f"✓ Created {len(all_chunks)} chunks from {len(documents)} documents")

    # Calculate average chunks per document
    avg_chunks = len(all_chunks) / len(documents) if documents else 0
    logger.info(f"  Average chunks per document: {avg_chunks:.1f}")

    # Step 4: Deduplicate
    logger.info(f"\n[4/5] Deduplicating (threshold={dedupe_threshold})...")
    deduplicator = Deduplicator(threshold=dedupe_threshold)

    deduplicated_chunks, duplicate_mapping = deduplicator.deduplicate(all_chunks)

    # Log deduplication stats
    dedup_stats = deduplicator.get_deduplication_stats(
        original_count=len(all_chunks),
        deduplicated_count=len(deduplicated_chunks),
        duplicate_mapping=duplicate_mapping,
    )

    logger.info(f"✓ Deduplication complete:")
    logger.info(f"  Original chunks: {dedup_stats['original_chunks']}")
    logger.info(f"  Deduplicated chunks: {dedup_stats['deduplicated_chunks']}")
    logger.info(f"  Duplicates removed: {dedup_stats['duplicates_removed']}")
    logger.info(f"  Removal rate: {dedup_stats['removal_rate']:.1%}")

    # Step 5: Generate embeddings and index
    logger.info(f"\n[5/5] Generating embeddings and indexing...")
    with Indexer(artifacts_dir=artifacts_dir) as indexer:
        indexer.index_chunks(deduplicated_chunks, batch_size=batch_size)

        # Get collection stats
        stats = indexer.get_collection_stats()
        logger.info(f"✓ Indexing complete:")
        logger.info(f"  Total chunks in MongoDB: {stats.get('total_chunks', 0)}")

        if "doc_type_distribution" in stats:
            logger.info("  Doc type distribution:")
            for doc_type, count in sorted(
                stats["doc_type_distribution"].items(), key=lambda x: -x[1]
            ):
                logger.info(f"    {doc_type}: {count}")

        if "source_distribution" in stats:
            logger.info("  Source distribution:")
            for source, count in sorted(
                stats["source_distribution"].items(), key=lambda x: -x[1]
            ):
                logger.info(f"    {source}: {count}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Ingestion Complete!")
    logger.info("=" * 80)
    logger.info(f"✓ Processed {len(documents)} documents")
    logger.info(f"✓ Created {len(deduplicated_chunks)} unique chunks")
    logger.info(f"✓ Stored in MongoDB: {config.db_name}.{config.collection_name}")
    logger.info(f"✓ Cached metadata: {artifacts_dir}/chunk_metadata.jsonl")
    logger.info("\nNext steps:")
    logger.info("  1. Test retrieval: python -m cli.query \"your query here\"")
    logger.info("  2. Run evaluation: python -m cli.evaluate")


if __name__ == "__main__":
    from src.utils.config import config

    # Validate configuration
    try:
        config.validate_required()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please set required environment variables in .env file")
        exit(1)

    ingest()
