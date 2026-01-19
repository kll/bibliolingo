"""LLM-based answer generation with OpenAI for BiblioLingo."""

import logging
from typing import List, Dict
from langchain_core.documents import Document
from openai import OpenAI

from src.utils.config import config
from src.retrieval.citation_formatter import CitationFormatter

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates answers using OpenAI with inline citations."""

    SYSTEM_PROMPT = """You are a helpful assistant that answers questions about internal technical documentation.

Your task is to provide accurate, concise answers based on the retrieved documents provided below.

Rules:
1. Answer the question directly and concisely
2. Base your answer ONLY on the information in the retrieved documents
3. If the documents don't contain enough information, say so
4. Include inline citations using [Document N] format when referencing specific information
5. Prefer information from ADRs and GitHub docs when multiple sources are available
6. If the question asks about decisions or architecture, prioritize ADR content

Format your answer in clear, professional language appropriate for technical documentation."""

    def __init__(
        self,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ):
        """
        Initialize answer generator.

        Args:
            model: OpenAI model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model or config.default_llm_model
        self.temperature = temperature if temperature is not None else config.llm_temperature
        self.max_tokens = max_tokens if max_tokens is not None else config.llm_max_tokens

        # Initialize OpenAI client
        self.client = OpenAI(api_key=config.openai_api_key)

        # Initialize citation formatter
        self.formatter = CitationFormatter()

    def generate_answer(
        self, query: str, documents: List[Document]
    ) -> Dict[str, any]:
        """
        Generate an answer with citations.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            Dictionary with 'answer' and 'citations_used'
        """
        if not documents:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "citations_used": [],
            }

        logger.info(f"Generating answer for query: {query}")

        # Format context for LLM
        context = self.formatter.format_for_llm(documents)

        # Build messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Retrieved Documents:\n\n{context}\n\nQuestion: {query}\n\nAnswer:",
            },
        ]

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content.strip()

            logger.info("Answer generated successfully")
            logger.debug(f"Model: {self.model}, Tokens used: {response.usage.total_tokens}")

            # Extract citation numbers from answer
            citations_used = self._extract_citations_from_answer(answer, documents)

            return {
                "answer": answer,
                "citations_used": citations_used,
                "model": self.model,
                "tokens_used": response.usage.total_tokens,
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "citations_used": [],
            }

    def _extract_citations_from_answer(
        self, answer: str, documents: List[Document]
    ) -> List[str]:
        """
        Extract citation references from the generated answer.

        Args:
            answer: Generated answer text
            documents: Original documents

        Returns:
            List of citation strings
        """
        import re

        citations = []
        citation_pattern = r'\[Document (\d+)\]'

        matches = re.findall(citation_pattern, answer)
        used_indices = set(int(m) for m in matches)

        # Create inline citations for referenced documents
        inline_citations = self.formatter.create_inline_citations(documents)

        for idx in sorted(used_indices):
            if 1 <= idx <= len(inline_citations):
                citations.append(inline_citations[idx - 1])

        return citations
