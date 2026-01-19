"""Hybrid retrieval combining BM25 and vector search with RRF fusion."""

import logging
from typing import List, Dict, Optional
from collections import defaultdict
import pickle
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.retrievers import BM25Retriever
from pymongo import MongoClient

from src.utils.config import config

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines BM25 keyword search with vector semantic search using RRF."""

    def __init__(
        self,
        alpha: float = None,
        top_k: int = None,
        rrf_k: int = None,
        artifacts_dir: str = "./artifacts",
    ):
        """
        Initialize hybrid retriever.

        Args:
            alpha: BM25 weight (0=vector only, 1=BM25 only)
            top_k: Number of results to return
            rrf_k: RRF constant for score calculation
            artifacts_dir: Directory for caching BM25 index
        """
        self.alpha = alpha if alpha is not None else config.default_alpha
        self.top_k = top_k if top_k is not None else config.default_top_k
        self.rrf_k = rrf_k if rrf_k is not None else config.rrf_k
        self.artifacts_dir = Path(artifacts_dir)

        # Initialize MongoDB connection
        self.mongo_client = MongoClient(config.mongo_db_url)
        self.collection = self.mongo_client[config.db_name][config.collection_name]

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=config.openai_api_key,
        )

        # Initialize vector store
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name="vector_index",
            text_key="content",
            embedding_key="embedding",
        )

        # BM25 retriever (loaded lazily)
        self._bm25_retriever = None
        self._documents_cache = None

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        alpha: Optional[float] = None,
        filters: Optional[Dict] = None,
    ) -> List[Document]:
        """
        Perform hybrid retrieval with RRF fusion.

        Args:
            query: Search query
            k: Number of results (defaults to self.top_k)
            alpha: BM25 weight override
            filters: MongoDB pre-filters for vector search

        Returns:
            List of Document objects ranked by hybrid score
        """
        k = k if k is not None else self.top_k
        alpha = alpha if alpha is not None else self.alpha

        logger.info(f"Hybrid retrieval: query='{query}', k={k}, alpha={alpha}")

        # Calculate candidate counts (retrieve more to improve fusion)
        bm25_k = k * 3
        vector_k = k * 3

        # BM25 search
        bm25_results = self._bm25_search(query, bm25_k)
        logger.debug(f"BM25 returned {len(bm25_results)} results")

        # Vector search with optional filters
        vector_results = self._vector_search(query, vector_k, filters)
        logger.debug(f"Vector search returned {len(vector_results)} results")

        # Combine with RRF
        combined = self._reciprocal_rank_fusion(
            result_lists=[bm25_results, vector_results],
            weights=[alpha, 1.0 - alpha],
        )

        # Return top-k
        return combined[:k]

    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """Perform BM25 keyword search."""
        # Load BM25 retriever
        if self._bm25_retriever is None:
            self._load_bm25_retriever()

        if self._bm25_retriever is None:
            logger.warning("BM25 retriever not available, returning empty results")
            return []

        try:
            results = self._bm25_retriever.invoke(query)
            return results[:k]
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []

    def _vector_search(
        self, query: str, k: int, filters: Optional[Dict] = None
    ) -> List[Document]:
        """Perform vector similarity search."""
        try:
            if filters:
                # Use pre-filter for vector search
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    pre_filter=filters,
                )
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                )
            return results
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[List[Document]],
        weights: List[float],
    ) -> List[Document]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.

        RRF Score = sum(weight_i / (k + rank_i)) for each list

        Args:
            result_lists: List of ranked document lists
            weights: Weight for each list

        Returns:
            Combined and reranked list of documents
        """
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)

        # Calculate RRF scores
        doc_scores = defaultdict(float)
        doc_map = {}  # Map content hash to document

        for list_idx, doc_list in enumerate(result_lists):
            weight = weights[list_idx]
            for rank, doc in enumerate(doc_list):
                # Use chunk_id from metadata as key for deduplication
                doc_key = doc.metadata.get("chunk_id", hash(doc.page_content))

                # RRF formula: weight / (k + rank + 1)
                rrf_score = weight / (self.rrf_k + rank + 1)
                doc_scores[doc_key] += rrf_score

                # Store document (keep first occurrence)
                if doc_key not in doc_map:
                    doc_map[doc_key] = doc

        # Sort by RRF score (descending)
        sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

        # Attach final scores to documents
        results = []
        for key in sorted_keys:
            doc = doc_map[key]
            doc.metadata["rrf_score"] = doc_scores[key]
            results.append(doc)

        return results

    def _load_bm25_retriever(self):
        """Load or create BM25 retriever."""
        bm25_path = self.artifacts_dir / "bm25_index.pkl"

        # Try to load from cache
        if bm25_path.exists():
            try:
                logger.info("Loading BM25 index from cache")
                with open(bm25_path, "rb") as f:
                    data = pickle.load(f)
                    self._documents_cache = data["documents"]
                    self._bm25_retriever = BM25Retriever.from_documents(
                        self._documents_cache
                    )
                logger.info(f"Loaded BM25 index with {len(self._documents_cache)} documents")
                return
            except Exception as e:
                logger.warning(f"Failed to load BM25 cache: {e}")

        # Load documents from MongoDB
        logger.info("Building BM25 index from MongoDB")
        documents = self._load_documents_from_mongodb()

        if not documents:
            logger.error("No documents found in MongoDB")
            return

        # Create BM25 retriever
        self._documents_cache = documents
        self._bm25_retriever = BM25Retriever.from_documents(documents)

        # Cache for future use
        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            with open(bm25_path, "wb") as f:
                pickle.dump({"documents": documents}, f)
            logger.info(f"Cached BM25 index with {len(documents)} documents")
        except Exception as e:
            logger.warning(f"Failed to cache BM25 index: {e}")

    def _load_documents_from_mongodb(self) -> List[Document]:
        """Load documents from MongoDB for BM25 indexing."""
        documents = []

        try:
            cursor = self.collection.find()
            for doc in cursor:
                # Extract fields
                content = doc.get("content", "")
                if not content:
                    continue

                # Build metadata
                metadata = {
                    "chunk_id": doc.get("chunk_id", ""),
                    "doc_id": doc.get("doc_id", ""),
                    "doc_title": doc.get("doc_title", ""),
                    "doc_type": doc.get("doc_type", ""),
                    "source": doc.get("source", ""),
                    "source_path": doc.get("source_path", ""),
                    "section_heading": doc.get("section_heading", ""),
                    "section_hierarchy": doc.get("section_hierarchy", []),
                    "section_type": doc.get("section_type"),
                    "component_tags": doc.get("component_tags", []),
                    "created_at": doc.get("created_at"),
                    "updated_at": doc.get("updated_at"),
                    "priority_score": doc.get("priority_score", 5),
                }

                documents.append(Document(page_content=content, metadata=metadata))

            logger.info(f"Loaded {len(documents)} documents from MongoDB")
        except Exception as e:
            logger.error(f"Error loading documents from MongoDB: {e}")

        return documents

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
