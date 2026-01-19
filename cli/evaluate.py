"""Evaluation CLI for BiblioLingo."""

import logging
import click
from datetime import datetime

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.score_normalizer import ScoreNormalizer
from src.evaluation.eval_runner import EvaluationRunner
from src.utils.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--gold-path",
    default="./eval/gold.jsonl",
    help="Path to gold evaluation dataset",
    type=click.Path(exists=True),
)
@click.option(
    "--output-path",
    default=None,
    help="Path to save evaluation report (default: ./artifacts/eval_report_TIMESTAMP.json)",
    type=click.Path(),
)
@click.option(
    "--k-values",
    default="1,3,5,10",
    help="Comma-separated k values to evaluate",
    type=str,
)
@click.option(
    "--alpha",
    default=None,
    help=f"BM25 weight override (default: {config.default_alpha})",
    type=float,
)
def evaluate(gold_path: str, output_path: str, k_values: str, alpha: float):
    """
    Evaluate BiblioLingo retrieval on gold dataset.

    This command:
    1. Loads gold evaluation queries
    2. Runs hybrid retrieval for each query
    3. Calculates Recall@k, MRR, and Precision@k metrics
    4. Saves detailed report to artifacts/

    Example:

        python -m cli.evaluate --gold-path eval/gold.jsonl
    """
    logger.info("=" * 80)
    logger.info("BiblioLingo Retrieval Evaluation")
    logger.info("=" * 80)

    # Generate timestamped output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./artifacts/eval_report_{timestamp}.json"

    # Parse k values
    k_list = [int(k.strip()) for k in k_values.split(",")]
    logger.info(f"K values: {k_list}")

    # Initialize components
    logger.info("Initializing retrieval components...")
    with HybridRetriever(alpha=alpha) as retriever:
        # Pass retriever params to normalizer for score normalization
        normalizer = ScoreNormalizer(
            rrf_k=retriever.rrf_k,
            alpha=retriever.alpha,
        )

        # Initialize evaluation runner
        runner = EvaluationRunner(k_values=k_list)

        # Load gold dataset
        logger.info(f"Loading gold dataset from {gold_path}")
        gold_queries = runner.load_gold_dataset(gold_path)

        # Run evaluation
        logger.info("\nStarting evaluation...")
        results = runner.evaluate(
            gold_queries=gold_queries,
            retriever=retriever,
            normalizer=normalizer,
        )

        # Save report
        runner.save_report(results, output_path)

        # Print summary
        runner.print_summary(results)

    # Success message
    logger.info(f"\n✓ Evaluation complete! Report saved to {output_path}")


if __name__ == "__main__":
    # Validate configuration
    try:
        config.validate_required()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please set required environment variables in .env file")
        exit(1)

    evaluate()
