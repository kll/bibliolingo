"""Citation formatter for BiblioLingo retrieval results."""

import logging
import re
from typing import List, Dict
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CitationFormatter:
    """Formats citations with minimal text snippets to avoid exposing full content."""

    def __init__(self, max_snippet_length: int = 150):
        """
        Initialize citation formatter.

        Args:
            max_snippet_length: Maximum length of content snippet
        """
        self.max_snippet_length = max_snippet_length

    def format_citations(self, documents: List[Document], query: str) -> Dict:
        """
        Format retrieved documents into citations.

        Args:
            documents: List of retrieved Document objects
            query: Original search query

        Returns:
            Dictionary with query, num_results, and citations list
        """
        citations = []

        for rank, doc in enumerate(documents, 1):
            citation = {
                "rank": rank,
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "doc_title": doc.metadata.get("doc_title", "Unknown"),
                "doc_type": doc.metadata.get("doc_type", "UNKNOWN"),
                "source": doc.metadata.get("source", "unknown"),
                "source_path": doc.metadata.get("source_path", ""),
                "section_heading": doc.metadata.get("section_heading", ""),
                "section_hierarchy": doc.metadata.get("section_hierarchy", []),
                "section_type": doc.metadata.get("section_type"),
                "updated_at": doc.metadata.get("updated_at"),
                "relevance_score": round(doc.metadata.get("final_score", 0.0), 4),
                "snippet": self._extract_snippet(doc.page_content, query),
            }
            citations.append(citation)

        return {
            "query": query,
            "num_results": len(documents),
            "citations": citations,
        }

    def format_for_display(self, citations_data: Dict) -> str:
        """
        Format citations for terminal display.

        Args:
            citations_data: Output from format_citations()

        Returns:
            Formatted string for display
        """
        lines = []
        lines.append(f"\nQuery: {citations_data['query']}")
        lines.append(f"Results: {citations_data['num_results']}")
        lines.append("=" * 80)

        for citation in citations_data["citations"]:
            lines.append(f"\n[{citation['rank']}] {citation['doc_title']}")
            lines.append(f"    Type: {citation['doc_type']} | Source: {citation['source']}")
            lines.append(f"    Section: {citation['section_heading']}")
            lines.append(f"    Path: {citation['source_path']}")
            lines.append(f"    Score: {citation['relevance_score']:.4f}")
            lines.append(f"    Snippet: {citation['snippet']}")

        return "\n".join(lines)

    def format_for_llm(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string for LLM.

        Includes citations but minimal content to stay within token limits.

        Args:
            documents: List of retrieved Document objects

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source_path = doc.metadata.get("source_path", "Unknown")
            section = doc.metadata.get("section_heading", "")
            doc_type = doc.metadata.get("doc_type", "")

            # Include full content for LLM (it needs context)
            # But we'll add clear citation markers
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {source_path}\n"
                f"Type: {doc_type}\n"
                f"Section: {section}\n"
                f"Content:\n{doc.page_content}\n"
            )

        return "\n---\n".join(context_parts)

    def _extract_snippet(self, text: str, query: str) -> str:
        """
        Extract a relevant snippet around query terms.

        Args:
            text: Full text content
            query: Search query

        Returns:
            Snippet string with ellipsis if truncated
        """
        if len(text) <= self.max_snippet_length:
            return text

        # Tokenize query
        query_terms = [t.lower() for t in re.split(r'\W+', query) if len(t) > 2]
        text_lower = text.lower()

        # Find best position (where most query terms appear)
        best_pos = 0
        max_term_count = 0

        # Sliding window to find best match area
        window_size = self.max_snippet_length
        for start in range(0, len(text) - window_size + 1, window_size // 2):
            window = text_lower[start : start + window_size]
            term_count = sum(1 for term in query_terms if term in window)
            if term_count > max_term_count:
                max_term_count = term_count
                best_pos = start

        # If no query terms found, just take the beginning
        if max_term_count == 0:
            snippet = text[: self.max_snippet_length]
            return snippet + "..." if len(text) > self.max_snippet_length else snippet

        # Extract snippet around best position
        start = max(0, best_pos - self.max_snippet_length // 4)
        end = min(len(text), best_pos + self.max_snippet_length)

        snippet = text[start:end].strip()

        # Add ellipsis if truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet

    def create_inline_citations(self, documents: List[Document]) -> List[str]:
        """
        Create compact inline citation strings.

        Useful for embedding citations in generated answers.

        Args:
            documents: List of Document objects

        Returns:
            List of citation strings like "[1: ADR - API Gateway, confluence/ADRs/...]"
        """
        citations = []

        for i, doc in enumerate(documents, 1):
            doc_title = doc.metadata.get("doc_title", "Unknown")
            source_path = doc.metadata.get("source_path", "")

            # Shorten path for readability
            short_path = source_path
            if len(short_path) > 40:
                parts = short_path.split("/")
                short_path = "/".join([parts[0], "...", parts[-1]])

            citation = f"[{i}: {doc_title}, {short_path}]"
            citations.append(citation)

        return citations
