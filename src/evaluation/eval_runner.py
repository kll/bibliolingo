"""Evaluation runner for BiblioLingo."""

import logging
import json
from typing import List, Dict
from pathlib import Path

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.score_normalizer import ScoreNormalizer
from src.evaluation.metrics import RetrievalMetrics

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Runs evaluation on a gold dataset and calculates metrics."""

    def __init__(self, k_values: List[int] = None):
        """
        Initialize evaluation runner.

        Args:
            k_values: List of k values to evaluate (default: [1, 3, 5, 10])
        """
        self.k_values = k_values or [1, 3, 5, 10]
        self.metrics_calculator = RetrievalMetrics()

    def load_gold_dataset(self, gold_path: str) -> List[Dict]:
        """
        Load gold evaluation dataset from JSONL file.

        Args:
            gold_path: Path to gold.jsonl file

        Returns:
            List of gold query dictionaries
        """
        gold_path = Path(gold_path)
        if not gold_path.exists():
            raise FileNotFoundError(f"Gold dataset not found: {gold_path}")

        gold_queries = []
        with open(gold_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    query_data = json.loads(line)
                    gold_queries.append(query_data)

        logger.info(f"Loaded {len(gold_queries)} queries from {gold_path}")
        return gold_queries

    def evaluate(
        self,
        gold_queries: List[Dict],
        retriever: HybridRetriever,
        normalizer: ScoreNormalizer,
    ) -> Dict:
        """
        Run evaluation on gold dataset.

        Args:
            gold_queries: List of gold query dictionaries
            retriever: HybridRetriever instance
            normalizer: ScoreNormalizer instance

        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Running evaluation on {len(gold_queries)} queries")
        logger.info(f"K values: {self.k_values}")

        query_results = []

        for i, gold_query in enumerate(gold_queries, 1):
            query = gold_query["query"]
            target_docs = gold_query["target_docs"]

            logger.info(f"[{i}/{len(gold_queries)}] Evaluating: {query}")

            # Retrieve documents
            max_k = max(self.k_values)
            retrieved_docs = retriever.retrieve(query=query, k=max_k)

            # Normalize scores and sort
            reranked_docs = normalizer.rerank(retrieved_docs)

            # Extract doc IDs (handle both chunk_id and doc_id fields)
            retrieved_doc_ids = []
            for doc in reranked_docs:
                # Try doc_id first (parent document)
                doc_id = doc.metadata.get("doc_id")
                if not doc_id:
                    # Fall back to chunk_id
                    doc_id = doc.metadata.get("chunk_id", "unknown")
                retrieved_doc_ids.append(doc_id)

            # Calculate metrics for each k
            result = {
                "query": query,
                "target_docs": target_docs,
                "retrieved_docs": retrieved_doc_ids[:max_k],
            }

            # Recall@k
            for k in self.k_values:
                recall = self.metrics_calculator.recall_at_k(
                    retrieved_doc_ids, target_docs, k
                )
                result[f"recall@{k}"] = recall

            # MRR (only needs to be calculated once)
            mrr = self.metrics_calculator.mean_reciprocal_rank(retrieved_doc_ids, target_docs)
            result["mrr"] = mrr

            # Precision@k
            for k in self.k_values:
                precision = self.metrics_calculator.precision_at_k(
                    retrieved_doc_ids, target_docs, k
                )
                result[f"precision@{k}"] = precision

            query_results.append(result)

            # Log per-query results
            logger.debug(f"  Recall@5: {result.get('recall@5', 0):.2f}")
            logger.debug(f"  MRR: {mrr:.3f}")

        # Calculate aggregate metrics
        aggregate_metrics = self.metrics_calculator.calculate_aggregate_metrics(query_results)

        logger.info("\nAggregate Metrics:")
        for metric_name, value in sorted(aggregate_metrics.items()):
            logger.info(f"  {metric_name}: {value:.4f}")

        return {
            "k_values": self.k_values,
            "num_queries": len(gold_queries),
            "aggregate_metrics": aggregate_metrics,
            "query_results": query_results,
        }

    def save_report(self, results: Dict, output_path: str):
        """
        Save evaluation report to JSON file.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Evaluation report saved to {output_path}")

    def print_summary(self, results: Dict):
        """
        Print a human-readable summary of evaluation results.

        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nQueries evaluated: {results['num_queries']}")
        print(f"K values: {results['k_values']}")

        print("\nAggregate Metrics:")
        print("-" * 40)

        metrics = results["aggregate_metrics"]

        # Print Recall@k
        print("\nRecall@k (% of queries with ≥1 relevant doc in top-k):")
        for k in results["k_values"]:
            key = f"recall@{k}"
            if key in metrics:
                value = metrics[key] * 100  # Convert to percentage
                print(f"  Recall@{k:2d}: {value:5.1f}%")

        # Print MRR
        if "mrr" in metrics:
            print(f"\nMRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")

        # Print Precision@k
        print("\nPrecision@k (fraction of top-k that are relevant):")
        for k in results["k_values"]:
            key = f"precision@{k}"
            if key in metrics:
                print(f"  Precision@{k:2d}: {metrics[key]:.4f}")

        print("\n" + "=" * 80)
