"""Query CLI for BiblioLingo."""

import logging
import click
import json

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.score_normalizer import ScoreNormalizer
from src.retrieval.citation_formatter import CitationFormatter
from src.generation.answer_generator import AnswerGenerator
from src.utils.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_filters(doc_type: str = None, source: str = None) -> dict:
    """
    Build MongoDB pre-filters for vector search.

    Args:
        doc_type: Document type filter
        source: Source filter (confluence/github)

    Returns:
        MongoDB filter dictionary
    """
    filters = {}

    if doc_type:
        filters["doc_type"] = doc_type

    if source:
        filters["source"] = source

    return filters if filters else None


@click.command()
@click.argument("query", type=str)
@click.option(
    "--k",
    default=None,
    help=f"Number of results to return (default: {config.default_top_k})",
    type=int,
)
@click.option(
    "--alpha",
    default=None,
    help=f"BM25 weight 0-1 (default: {config.default_alpha})",
    type=float,
)
@click.option(
    "--doc-type",
    default=None,
    help="Filter by document type (ADR, RFC, DESIGN, HOWTO, etc.)",
    type=str,
)
@click.option(
    "--source",
    default=None,
    help="Filter by source (confluence, github)",
    type=click.Choice(["confluence", "github"], case_sensitive=False),
)
@click.option(
    "--generate-answer",
    is_flag=True,
    help="Generate LLM answer with citations",
)
@click.option(
    "--json-output",
    is_flag=True,
    help="Output results as JSON",
)
def query(
    query: str,
    k: int,
    alpha: float,
    doc_type: str,
    source: str,
    generate_answer: bool,
    json_output: bool,
):
    """
    Query the BiblioLingo RAG system.

    QUERY: Search query text

    Examples:

        python -m cli.query "How do feature flags work?"

        python -m cli.query "API gateway selection" --doc-type ADR --k 5

        python -m cli.query "YARP configuration" --alpha 0.7 --generate-answer
    """
    logger.info(f"Query: {query}")

    # Build filters
    filters = build_filters(doc_type=doc_type, source=source)

    # Initialize components
    with HybridRetriever(alpha=alpha, top_k=k) as retriever:
        # Pass retriever params to normalizer for score normalization
        normalizer = ScoreNormalizer(
            rrf_k=retriever.rrf_k,
            alpha=retriever.alpha,
        )
        formatter = CitationFormatter()

        # Perform hybrid retrieval
        logger.info("Performing hybrid retrieval...")
        documents = retriever.retrieve(query=query, filters=filters)

        if not documents:
            click.echo("\nNo results found.")
            return

        logger.info(f"Retrieved {len(documents)} documents")

        # Normalize scores and sort by relevance
        logger.info("Normalizing scores...")
        reranked_docs = normalizer.rerank(documents)

        # Check confidence
        is_confident, top_score = normalizer.check_confidence(reranked_docs)

        if not is_confident and filters:
            # Try fallback with relaxed filters
            logger.warning("Low confidence, attempting fallback with relaxed filters")
            relaxed_filters = normalizer.suggest_fallback_filters(filters, top_score)

            fallback_docs = retriever.retrieve(query=query, filters=relaxed_filters)
            if fallback_docs:
                reranked_docs = normalizer.rerank(fallback_docs)
                is_confident, top_score = normalizer.check_confidence(reranked_docs)

        # Format citations
        citations_data = formatter.format_citations(reranked_docs, query)

        # Output results
        if json_output:
            # JSON output
            output = {"citations": citations_data}

            if generate_answer:
                # Generate answer
                answer_gen = AnswerGenerator()
                answer_data = answer_gen.generate_answer(query, reranked_docs)
                output["answer"] = answer_data

            click.echo(json.dumps(output, indent=2))

        else:
            # Terminal output
            display_text = formatter.format_for_display(citations_data)
            click.echo(display_text)

            if generate_answer:
                # Generate answer
                click.echo("\n" + "=" * 80)
                click.echo("Generated Answer")
                click.echo("=" * 80)

                answer_gen = AnswerGenerator()
                answer_data = answer_gen.generate_answer(query, reranked_docs)

                click.echo(f"\n{answer_data['answer']}\n")

                if answer_data.get("citations_used"):
                    click.echo("Citations:")
                    for citation in answer_data["citations_used"]:
                        click.echo(f"  {citation}")

        # Show confidence warning if needed
        if not is_confident:
            click.echo(
                f"\n⚠️  Low confidence (score: {top_score:.4f}). "
                "Results may not be highly relevant."
            )


if __name__ == "__main__":
    # Validate configuration
    try:
        config.validate_required()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please set required environment variables in .env file")
        exit(1)

    query()
