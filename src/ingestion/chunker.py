"""Markdown-aware chunking with ADR section detection for BiblioLingo."""

import logging
import re
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a markdown section."""
    level: int  # Heading level (1-6)
    heading: str  # Heading text
    content: str  # Section content (excluding heading)
    line_start: int  # Starting line number
    line_end: int  # Ending line number


@dataclass
class Chunk:
    """Represents a chunk of content with metadata."""
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    source: str
    source_path: str
    section_heading: str
    section_hierarchy: List[str]
    section_type: Optional[str]  # For ADRs: context|decision|consequences|alternatives
    component_tags: List[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    content: str
    content_hash: str
    char_count: int
    priority_score: int


class MarkdownChunker:
    """Chunks markdown documents by headings with ADR special handling."""

    # ADR section patterns
    ADR_SECTION_PATTERNS = {
        "context": [
            r"^context$",
            r"^context\s+and\s+problem\s+statement$",
            r"^problem\s+statement$",
            r"^background$",
        ],
        "decision": [
            r"^decision$",
            r"^decision\s+outcome$",
            r"^chosen\s+option$",
            r"^solution$",
        ],
        "consequences": [
            r"^consequences$",
            r"^pros\s+and\s+cons$",
            r"^trade[-\s]?offs$",
            r"^implications$",
        ],
        "alternatives": [
            r"^considered\s+options?$",
            r"^alternatives?$",
            r"^decision\s+drivers?$",
            r"^options\s+considered$",
        ],
    }

    # Minimum and maximum chunk sizes
    MIN_CHUNK_SIZE = 50  # chars
    MAX_CHUNK_SIZE = 2000  # chars

    def chunk_document(self, doc, metadata) -> List[Chunk]:
        """
        Chunk a markdown document by headings.

        Args:
            doc: MarkdownDocument object
            metadata: DocumentMetadata object

        Returns:
            List of Chunk objects
        """
        content = doc.content

        # Parse markdown into sections
        sections = self._parse_markdown_sections(content)

        if not sections:
            # No headings found, treat entire document as one chunk
            logger.warning(f"No headings found in {metadata.doc_id}, creating single chunk")
            return [self._create_chunk_from_content(content, doc, metadata, chunk_index=0)]

        # Generate chunks from sections
        chunks = []

        if metadata.doc_type == "ADR":
            # Special ADR handling
            chunks = self._chunk_adr(sections, doc, metadata)
        else:
            # Standard chunking by headings
            chunks = self._chunk_by_headings(sections, doc, metadata)

        # Filter out chunks that are too small
        chunks = [c for c in chunks if c.char_count >= self.MIN_CHUNK_SIZE]

        logger.debug(f"Created {len(chunks)} chunks from {metadata.doc_id}")
        return chunks

    def _parse_markdown_sections(self, content: str) -> List[Section]:
        """Parse markdown content into hierarchical sections."""
        lines = content.split("\n")
        sections = []
        current_section = None

        for i, line in enumerate(lines):
            # Check if line is a heading (# Heading or === or ---)
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if heading_match:
                # Save previous section
                if current_section:
                    current_section.line_end = i - 1
                    sections.append(current_section)

                # Start new section
                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()

                current_section = Section(
                    level=level,
                    heading=heading,
                    content="",
                    line_start=i + 1,
                    line_end=len(lines) - 1,
                )
            elif current_section:
                # Add content to current section
                current_section.content += line + "\n"

        # Add last section
        if current_section:
            sections.append(current_section)

        return sections

    def _chunk_adr(self, sections: List[Section], doc, metadata) -> List[Chunk]:
        """Chunk ADR with section type classification."""
        chunks = []
        section_hierarchy = []
        chunk_index = 0

        for section in sections:
            # Update hierarchy based on section level
            if section.level <= len(section_hierarchy):
                section_hierarchy = section_hierarchy[: section.level - 1]
            section_hierarchy.append(section.heading)

            # Determine section type
            section_type = self._classify_adr_section(section.heading)

            # Create chunk
            chunk = self._create_chunk(
                content=section.content.strip(),
                doc=doc,
                metadata=metadata,
                chunk_index=chunk_index,
                section_heading=section.heading,
                section_hierarchy=list(section_hierarchy),
                section_type=section_type,
            )

            if chunk:
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _chunk_by_headings(self, sections: List[Section], doc, metadata) -> List[Chunk]:
        """Standard chunking by markdown headings."""
        chunks = []
        section_hierarchy = []
        chunk_index = 0

        for section in sections:
            # Update hierarchy based on section level
            if section.level <= len(section_hierarchy):
                section_hierarchy = section_hierarchy[: section.level - 1]
            section_hierarchy.append(section.heading)

            content = section.content.strip()

            # If section is too large, split it further
            if len(content) > self.MAX_CHUNK_SIZE:
                sub_chunks = self._split_large_section(content, self.MAX_CHUNK_SIZE)
                for sub_index, sub_content in enumerate(sub_chunks):
                    chunk = self._create_chunk(
                        content=sub_content,
                        doc=doc,
                        metadata=metadata,
                        chunk_index=chunk_index,
                        section_heading=f"{section.heading} (part {sub_index + 1})",
                        section_hierarchy=list(section_hierarchy),
                        section_type=None,
                    )
                    if chunk:
                        chunks.append(chunk)
                        chunk_index += 1
            else:
                # Create single chunk for section
                chunk = self._create_chunk(
                    content=content,
                    doc=doc,
                    metadata=metadata,
                    chunk_index=chunk_index,
                    section_heading=section.heading,
                    section_hierarchy=list(section_hierarchy),
                    section_type=None,
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1

        return chunks

    def _classify_adr_section(self, heading: str) -> Optional[str]:
        """Classify an ADR section heading into a section type."""
        heading_lower = heading.lower().strip()

        for section_type, patterns in self.ADR_SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, heading_lower, re.IGNORECASE):
                    return section_type

        return None

    def _create_chunk(
        self,
        content: str,
        doc,
        metadata,
        chunk_index: int,
        section_heading: str,
        section_hierarchy: List[str],
        section_type: Optional[str],
    ) -> Optional[Chunk]:
        """Create a chunk object with full metadata."""
        if len(content.strip()) < self.MIN_CHUNK_SIZE:
            return None

        # Generate chunk ID
        chunk_id = f"{metadata.doc_id}-{section_heading.lower().replace(' ', '-')[:30]}-{chunk_index}"
        # Sanitize chunk_id
        chunk_id = re.sub(r'[^a-z0-9\-]', '', chunk_id)

        # Generate content hash for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=metadata.doc_id,
            doc_title=metadata.doc_title,
            doc_type=metadata.doc_type,
            source=metadata.source,
            source_path=metadata.source_path,
            section_heading=section_heading,
            section_hierarchy=section_hierarchy,
            section_type=section_type,
            component_tags=metadata.component_tags,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
            content=content.strip(),
            content_hash=content_hash,
            char_count=len(content.strip()),
            priority_score=metadata.priority_score,
        )

        return chunk

    def _create_chunk_from_content(
        self, content: str, doc, metadata, chunk_index: int
    ) -> Chunk:
        """Create a chunk from raw content (no sections found)."""
        content = content.strip()

        # Generate chunk ID
        chunk_id = f"{metadata.doc_id}-full-{chunk_index}"

        # Generate content hash
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=metadata.doc_id,
            doc_title=metadata.doc_title,
            doc_type=metadata.doc_type,
            source=metadata.source,
            source_path=metadata.source_path,
            section_heading=metadata.doc_title,
            section_hierarchy=[metadata.doc_title],
            section_type=None,
            component_tags=metadata.component_tags,
            created_at=metadata.created_at,
            updated_at=metadata.updated_at,
            content=content,
            content_hash=content_hash,
            char_count=len(content),
            priority_score=metadata.priority_score,
        )

        return chunk

    def _split_large_section(self, content: str, max_size: int) -> List[str]:
        """Split a large section into smaller chunks."""
        # Try to split by paragraphs first
        paragraphs = content.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        # If still too large, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_size:
                # Simple sentence splitting
                sentences = re.split(r'([.!?]+\s+)', chunk)
                sub_chunk = ""
                for sentence in sentences:
                    if len(sub_chunk) + len(sentence) <= max_size:
                        sub_chunk += sentence
                    else:
                        if sub_chunk:
                            final_chunks.append(sub_chunk.strip())
                        sub_chunk = sentence
                if sub_chunk:
                    final_chunks.append(sub_chunk.strip())
            else:
                final_chunks.append(chunk)

        return final_chunks
