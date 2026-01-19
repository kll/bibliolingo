"""Indexer for generating embeddings and storing in MongoDB."""

import logging
import json
from typing import List
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pymongo import MongoClient
from pymongo.errors import OperationFailure

from src.utils.config import config

logger = logging.getLogger(__name__)


class Indexer:
    """Generates embeddings and stores chunks in MongoDB."""

    def __init__(self, artifacts_dir: str = "./artifacts"):
        """
        Initialize indexer.

        Args:
            artifacts_dir: Directory for caching metadata
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MongoDB
        self.mongo_client = MongoClient(config.mongo_db_url)
        self.db = self.mongo_client[config.db_name]
        self.collection = self.db[config.collection_name]

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=config.openai_api_key,
        )

    def index_chunks(self, chunks: List, batch_size: int = 100) -> None:
        """
        Generate embeddings and store chunks in MongoDB.

        Args:
            chunks: List of Chunk objects
            batch_size: Number of chunks to process in each batch
        """
        if not chunks:
            logger.warning("No chunks to index")
            return

        logger.info(f"Indexing {len(chunks)} chunks (batch_size={batch_size})")

        # Clear existing data
        self._clear_collection()

        # Process in batches
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for batch_idx in range(0, len(chunks), batch_size):
            batch = chunks[batch_idx : batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")

            # Generate embeddings for batch
            texts = [chunk.content for chunk in batch]
            try:
                embeddings = self.embeddings.embed_documents(texts)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {batch_num}: {e}")
                continue

            # Prepare documents for MongoDB
            documents = []
            for chunk, embedding in zip(batch, embeddings):
                doc = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "doc_type": chunk.doc_type,
                    "source": chunk.source,
                    "source_path": chunk.source_path,
                    "section_heading": chunk.section_heading,
                    "section_hierarchy": chunk.section_hierarchy,
                    "section_type": chunk.section_type,
                    "component_tags": chunk.component_tags,
                    "created_at": chunk.created_at,
                    "updated_at": chunk.updated_at,
                    "content": chunk.content,
                    "content_hash": chunk.content_hash,
                    "char_count": chunk.char_count,
                    "priority_score": chunk.priority_score,
                    "embedding": embedding,
                }
                documents.append(doc)

            # Insert into MongoDB
            try:
                result = self.collection.insert_many(documents)
                logger.debug(f"Inserted {len(result.inserted_ids)} documents")
            except Exception as e:
                logger.error(f"Error inserting batch {batch_num} into MongoDB: {e}")
                continue

        logger.info(f"Indexing complete: {len(chunks)} chunks stored in MongoDB")

        # Create vector search index if it doesn't exist
        self._ensure_vector_index()

        # Cache chunk metadata (without embeddings and full content)
        self._cache_metadata(chunks)

    def _clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            count = self.collection.count_documents({})
            if count > 0:
                logger.info(f"Clearing {count} existing documents from collection")
                self.collection.delete_many({})
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

    def _ensure_vector_index(self) -> None:
        """Ensure vector search index exists on the collection."""
        try:
            # Check if index already exists
            indexes = list(self.collection.list_indexes())
            index_names = [idx['name'] for idx in indexes]

            if "vector_index" in index_names:
                logger.info("Vector index already exists")
                return

            # Create vector search index
            # Note: For MongoDB Atlas Local, we need to use the createSearchIndex command
            logger.info("Creating vector search index...")

            # Atlas Search Index definition
            index_definition = {
                "name": "vector_index",
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": 1536,  # OpenAI text-embedding-3-small
                            "similarity": "cosine",
                        }
                    ]
                },
            }

            # Try to create the index using MongoDB Atlas Local commands
            try:
                self.db.command("createSearchIndex", config.collection_name, **index_definition)
                logger.info("Vector search index created successfully")
            except OperationFailure as e:
                # MongoDB Atlas Local might not support createSearchIndex command yet
                # In that case, we'll log a warning but continue
                logger.warning(
                    f"Could not create vector search index automatically: {e}. "
                    f"You may need to create it manually in Atlas or MongoDB Compass."
                )

        except Exception as e:
            logger.warning(f"Error ensuring vector index: {e}")

    def _cache_metadata(self, chunks: List) -> None:
        """
        Cache chunk metadata to artifacts directory.

        Args:
            chunks: List of Chunk objects
        """
        metadata_path = self.artifacts_dir / "chunk_metadata.jsonl"

        try:
            with open(metadata_path, "w") as f:
                for chunk in chunks:
                    # Convert chunk to dict (without embedding)
                    metadata = {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "doc_title": chunk.doc_title,
                        "doc_type": chunk.doc_type,
                        "source": chunk.source,
                        "source_path": chunk.source_path,
                        "section_heading": chunk.section_heading,
                        "section_hierarchy": chunk.section_hierarchy,
                        "section_type": chunk.section_type,
                        "component_tags": chunk.component_tags,
                        "created_at": chunk.created_at,
                        "updated_at": chunk.updated_at,
                        "content_hash": chunk.content_hash,
                        "char_count": chunk.char_count,
                        "priority_score": chunk.priority_score,
                    }
                    f.write(json.dumps(metadata) + "\n")

            logger.info(f"Cached metadata for {len(chunks)} chunks to {metadata_path}")
        except Exception as e:
            logger.error(f"Error caching metadata: {e}")

    def get_collection_stats(self) -> dict:
        """Get statistics about the indexed collection."""
        try:
            count = self.collection.count_documents({})

            # Get distribution by doc_type
            doc_type_pipeline = [{"$group": {"_id": "$doc_type", "count": {"$sum": 1}}}]
            doc_type_dist = list(self.collection.aggregate(doc_type_pipeline))

            # Get distribution by source
            source_pipeline = [{"$group": {"_id": "$source", "count": {"$sum": 1}}}]
            source_dist = list(self.collection.aggregate(source_pipeline))

            stats = {
                "total_chunks": count,
                "doc_type_distribution": {item["_id"]: item["count"] for item in doc_type_dist},
                "source_distribution": {item["_id"]: item["count"] for item in source_dist},
            }

            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def close(self):
        """Close MongoDB connection."""
        if self.mongo_client:
            self.mongo_client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
