"""Benchmark suite for traffic events retrieval system"""

import time
from typing import List, Dict, Any
from .evaluator import EvaluationFramework
from .metrics import EvaluationMetrics


class BenchmarkSuite:
    """Benchmark suite for evaluating IR systems"""
    
    def __init__(self):
        self.evaluator = EvaluationFramework()
        self.metrics = EvaluationMetrics()
    
    def run_benchmark(self, 
                    queries: List[str],
                    qrels: Dict[str, Any],
                    k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Run complete benchmark suite
        
        Args:
            queries: List of queries to benchmark
            qrels: Relevance judgments
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with benchmark results
        """
        print("=" * 80)
        print("ROUTIQ IR - BENCHMARK SUITE")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run evaluation
        results = self.evaluator.evaluate_system(queries, qrels, k_values)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Add benchmark metadata
        benchmark_results = {
            'benchmark_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_queries': len(queries),
                'k_values': k_values,
                'total_time': total_time,
                'avg_time_per_query': total_time / len(queries)
            },
            'evaluation_results': results
        }
        
        # Print benchmark summary
        print(f"\nBenchmark completed in {total_time:.2f} seconds")
        print(f"Average time per query: {total_time / len(queries):.4f} seconds")
        
        return benchmark_results
    
    def compare_systems(self, 
                     system_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple IR systems
        
        Args:
            system_results: Dictionary mapping system names to their evaluation results
            
        Returns:
            Dictionary with comparison results
        """
        print("\n" + "=" * 80)
        print("SYSTEM COMPARISON")
        print("=" * 80)
        
        comparison = {}
        
        for system_name, results in system_results.items():
            if 'final_metrics' in results:
                metrics = results['final_metrics']
                comparison[system_name] = {
                    'map_score': metrics.get('map_score', 0.0),
                    'avg_precision': metrics.get('average_precision', 0.0),
                    'avg_precision_at_1': metrics.get('avg_precision_at_1', 0.0),
                    'avg_precision_at_3': metrics.get('avg_precision_at_3', 0.0),
                    'avg_precision_at_5': metrics.get('avg_precision_at_5', 0.0),
                    'avg_precision_at_10': metrics.get('avg_precision_at_10', 0.0),
                    'avg_recall_at_1': metrics.get('avg_recall_at_1', 0.0),
                    'avg_recall_at_3': metrics.get('avg_recall_at_3', 0.0),
                    'avg_recall_at_5': metrics.get('avg_recall_at_5', 0.0),
                    'avg_recall_at_10': metrics.get('avg_recall_at_10', 0.0)
                }
        
        # Print comparison table
        if comparison:
            print(f"\n{'System':<15} {'MAP':<10} {'P@1':<10} {'P@3':<10} {'P@5':<10} {'P@10':<10} {'R@1':<10} {'R@3':<10} {'R@5':<10} {'R@10':<10}")
            print("-" * 100)
            
            for system_name, metrics in comparison.items():
                print(f"{system_name:<15} {metrics['map_score']:<10.4f} {metrics['avg_precision_at_1']:<10.4f} {metrics['avg_precision_at_3']:<10.4f} {metrics['avg_precision_at_5']:<10.4f} {metrics['avg_precision_at_10']:<10.4f} {metrics['avg_recall_at_1']:<10.4f} {metrics['avg_recall_at_3']:<10.4f} {metrics['avg_recall_at_5']:<10.4f} {metrics['avg_recall_at_10']:<10.4f}")
        
        print("\n" + "=" * 80)
        
        return comparison
