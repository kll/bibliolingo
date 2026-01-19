"""Metadata extraction and normalization for BiblioLingo."""

import logging
import re
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentMetadata:
    """Normalized metadata for a document."""

    def __init__(self):
        self.doc_id: str = ""
        self.doc_title: str = ""
        self.doc_type: str = "UNKNOWN"
        self.source: str = ""
        self.source_path: str = ""
        self.created_at: Optional[str] = None
        self.updated_at: Optional[str] = None
        self.component_tags: List[str] = []
        self.priority_score: int = 5  # Default priority

    def __repr__(self) -> str:
        return f"DocumentMetadata(doc_id={self.doc_id}, doc_type={self.doc_type}, source={self.source})"


class MetadataNormalizer:
    """Extracts and normalizes metadata from markdown documents."""

    # Doc type detection patterns
    DOC_TYPE_PATTERNS = {
        "ADR": [
            r"^#\s*ADR[:\s-]",
            r"(?i)architecture\s+decision\s+record",
            r"^adr-",
            r"/ADRs/",
        ],
        "RFC": [r"^#\s*RFC[:\s-]", r"(?i)request\s+for\s+comments?"],
        "DESIGN": [r"^#\s*Design[:\s-]", r"(?i)design\s+document", r"(?i)design\s+doc"],
        "HOWTO": [
            r"^#\s*How\s+to",
            r"(?i)how[-\s]to",
            r"(?i)setup",
            r"(?i)getting\s+started",
            r"first-call",
            r"setup",
        ],
        "RUNBOOK": [r"^#\s*Runbook", r"(?i)runbook", r"(?i)playbook"],
        "POSTMORTEM": [r"(?i)post[-\s]?mortem", r"(?i)incident\s+report"],
    }

    # Priority scores by doc_type and source
    PRIORITY_SCORES = {
        ("ADR", "confluence"): 10,
        ("ADR", "github"): 10,
        ("RFC", "confluence"): 9,
        ("RFC", "github"): 9,
        ("DESIGN", "confluence"): 8,
        ("DESIGN", "github"): 9,
        ("HOWTO", "github"): 8,
        ("HOWTO", "confluence"): 6,
        ("RUNBOOK", "github"): 7,
        ("RUNBOOK", "confluence"): 6,
    }

    def normalize(self, doc, content: str = None) -> DocumentMetadata:
        """
        Extract and normalize metadata from a document.

        Args:
            doc: MarkdownDocument object
            content: Optional content override

        Returns:
            DocumentMetadata object
        """
        metadata = DocumentMetadata()

        # Use provided content or doc content
        content = content or doc.content

        # Basic source information
        metadata.source = doc.source or "unknown"
        metadata.source_path = doc.relative_path or str(doc.path)

        # Extract doc_id from filename
        filename = Path(doc.path).stem
        metadata.doc_id = self._extract_doc_id(filename, metadata.source)

        # Extract doc_title
        metadata.doc_title = self._extract_title(content, filename)

        # Determine doc_type
        metadata.doc_type = self._determine_doc_type(content, doc.path, doc.source_subdir)

        # Extract timestamps from Confluence metadata
        if metadata.source == "confluence":
            created, updated = self._extract_confluence_timestamps(content)
            metadata.created_at = created
            metadata.updated_at = updated

        # Extract component tags
        metadata.component_tags = self._extract_component_tags(content, filename)

        # Calculate priority score
        metadata.priority_score = self._calculate_priority(
            metadata.doc_type, metadata.source
        )

        return metadata

    def _extract_doc_id(self, filename: str, source: str) -> str:
        """Extract document ID from filename."""
        if source == "confluence":
            # Confluence files are named like "3074097170-adr-api-gateway-selection.md"
            match = re.match(r"(\d+)-(.+)", filename)
            if match:
                confluence_id = match.group(1)
                slug = match.group(2)
                # Use a more readable format
                return f"{source}-{confluence_id}"

        # For GitHub or unknown sources, use the filename
        return f"{source}-{filename}"

    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract document title from content."""
        # Try to find first H1 heading
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                # Remove markdown links
                title = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", title)
                return title

        # Fallback to filename
        return fallback.replace("-", " ").replace("_", " ").title()

    def _determine_doc_type(self, content: str, path: Path, source_subdir: str) -> str:
        """Determine document type from content and path."""
        # Check source directory first
        if source_subdir and "ADR" in source_subdir.upper():
            return "ADR"

        # Check filename
        filename = path.name.lower()
        for doc_type, patterns in self.DOC_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, filename):
                    return doc_type

        # Check content (first 500 chars)
        header = content[:500]
        for doc_type, patterns in self.DOC_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, header, re.MULTILINE | re.IGNORECASE):
                    return doc_type

        # Default to UNKNOWN
        return "UNKNOWN"

    def _extract_confluence_timestamps(self, content: str) -> tuple[Optional[str], Optional[str]]:
        """Extract Created and Updated timestamps from Confluence metadata."""
        created = None
        updated = None

        # Look for lines like:
        # **Created:** 2025-01-07 14:25:47 UTC
        # **Updated:** 2025-01-30 15:21:39 UTC
        created_match = re.search(r"\*\*Created:\*\*\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+UTC)", content)
        if created_match:
            created = created_match.group(1).replace(" ", "T").replace("UTC", "Z")

        updated_match = re.search(r"\*\*Updated:\*\*\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+UTC)", content)
        if updated_match:
            updated = updated_match.group(1).replace(" ", "T").replace("UTC", "Z")

        return created, updated

    def _extract_component_tags(self, content: str, filename: str) -> List[str]:
        """Extract component/technology tags from content and filename."""
        tags = set()

        # Common tech keywords to extract
        tech_keywords = [
            "api", "gateway", "yarp", "caddy", "authentication", "auth",
            "microservice", "mongo", "mongodb", "docker", "kubernetes", "k8s",
            "angular", "react", "vue", "grails", "nodejs", "python",
            "twilio", "sms", "calling", "notification", "event-bus",
            "masstransit", "orleans", "pwa", "migration", "infrastructure",
        ]

        # Check filename
        filename_lower = filename.lower()
        for keyword in tech_keywords:
            if keyword in filename_lower:
                tags.add(keyword)

        # Check content (case-insensitive)
        content_lower = content.lower()
        for keyword in tech_keywords:
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + re.escape(keyword) + r'\b', content_lower):
                tags.add(keyword)

        return sorted(list(tags))

    def _calculate_priority(self, doc_type: str, source: str) -> int:
        """Calculate priority score based on doc_type and source."""
        key = (doc_type, source)
        return self.PRIORITY_SCORES.get(key, 5)  # Default priority of 5
