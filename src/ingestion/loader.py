"""Markdown document loader for BiblioLingo."""

import logging
from pathlib import Path
from typing import List, Dict
import os

logger = logging.getLogger(__name__)


class MarkdownDocument:
    """Represents a loaded markdown document."""

    def __init__(self, path: Path, content: str):
        self.path = path
        self.content = content
        self.relative_path = None
        self.source = None  # Will be set to 'confluence' or 'github'
        self.source_subdir = None  # e.g., 'ADRs', 'DEV', 'bl-platform/docs'

    def __repr__(self) -> str:
        return f"MarkdownDocument(path={self.path}, source={self.source})"


class MarkdownLoader:
    """Loads markdown files from data directories."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

    def load_all(self) -> List[MarkdownDocument]:
        """Load all markdown files from data directory."""
        documents = []

        # Load Confluence documents
        confluence_dir = self.data_dir / "confluence"
        if confluence_dir.exists():
            for subdir in ["ADRs", "DEV"]:
                subdir_path = confluence_dir / subdir
                if subdir_path.exists():
                    docs = self._load_from_directory(
                        subdir_path, source="confluence", source_subdir=subdir
                    )
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} documents from confluence/{subdir}")

        # Load GitHub documents
        github_dir = self.data_dir / "github"
        if github_dir.exists():
            docs = self._load_from_directory(github_dir, source="github", source_subdir="github")
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} documents from github")

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def _load_from_directory(
        self, directory: Path, source: str, source_subdir: str
    ) -> List[MarkdownDocument]:
        """Load all markdown files from a directory recursively."""
        documents = []

        for md_file in directory.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                doc = MarkdownDocument(path=md_file, content=content)
                doc.source = source
                doc.source_subdir = source_subdir

                # Set relative path from data directory
                doc.relative_path = str(md_file.relative_to(self.data_dir))

                documents.append(doc)
            except Exception as e:
                logger.error(f"Error loading {md_file}: {e}")
                continue

        return documents

    def load_single(self, file_path: str) -> MarkdownDocument:
        """Load a single markdown file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        doc = MarkdownDocument(path=path, content=content)

        # Determine source from path
        try:
            rel_path = path.relative_to(self.data_dir)
            parts = rel_path.parts
            if len(parts) > 0:
                if parts[0] == "confluence":
                    doc.source = "confluence"
                    doc.source_subdir = parts[1] if len(parts) > 1 else "unknown"
                elif parts[0] == "github":
                    doc.source = "github"
                    doc.source_subdir = "github"
                doc.relative_path = str(rel_path)
        except ValueError:
            # File is outside data directory
            doc.source = "unknown"
            doc.relative_path = str(path)

        return doc
