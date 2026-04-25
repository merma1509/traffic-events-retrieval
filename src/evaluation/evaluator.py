"""Evaluation framework for traffic events retrieval system"""

import json
from typing import List, Dict, Any, Set, Optional
from datetime import datetime
import pandas as pd

from .metrics import EvaluationMetrics
from retrieval import RetrievalEngine
from indexing import IndexManager


class EvaluationFramework:
    """Complete evaluation framework for IR systems"""
    
    def __init__(self, indices_dir: str = "data/indices"):
        self.metrics_calculator = EvaluationMetrics()
        self.retrieval_engine = RetrievalEngine(indices_dir)
        self.index_manager = IndexManager(indices_dir)
        self.evaluation_results = []
        
    def load_qrels(self, qrels_file: str) -> Dict[str, Set[str]]:
        """
        Load relevance judgments (qrels)
        
        Args:
            qrels_file: Path to qrels file
            
        Returns:
            Dictionary mapping query IDs to sets of relevant document IDs
        """
        try:
            with open(qrels_file, 'r', encoding='utf-8') as f:
                qrels = json.load(f)
            
            # Convert to proper format
            formatted_qrels = {}
            for query_id, rel_docs in qrels.items():
                if isinstance(rel_docs, list):
                    formatted_qrels[query_id] = set(rel_docs)
                elif isinstance(rel_docs, dict):
                    # Handle different relevance levels
                    relevant_set = set()
                    for doc_id, relevance in rel_docs.items():
                        if relevance > 0:  # Non-zero relevance means relevant
                            relevant_set.add(doc_id)
                    formatted_qrels[query_id] = relevant_set
                else:
                    formatted_qrels[query_id] = set()
            
            print(f"Loaded {len(formatted_qrels)} query relevance judgments")
            return formatted_qrels
            
        except FileNotFoundError:
            print(f"Qrels file not found: {qrels_file}")
            return {}
        except Exception as e:
            print(f"Error loading qrels: {e}")
            return {}
    
    def create_sample_qrels(self, corpus: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        """Create sample relevance judgments for testing
        
        Args:
            corpus: List of documents to create qrels for
            
        Returns:
            Dictionary of sample qrels
        """
        sample_qrels = {}
        
        # Create sample queries and relevant documents
        sample_queries = [
            {
                'query': 'heavy congestion',
                'relevant_doc_ids': {'doc1', 'doc3'}        # Documents about heavy congestion
            },
            {
                'query': 'rain weather',
                'relevant_doc_ids': {'doc1', 'doc2', 'doc3'}  # Documents with rain
            },
            {
                'query': 'highway traffic',
                'relevant_doc_ids': {'doc1', 'doc4'}  # Documents about highways
            },
            {
                'query': 'rush hour',
                'relevant_doc_ids': {'doc2', 'doc5'}  # Documents about rush hour
            },
            {
                'query': 'accident',
                'relevant_doc_ids': {'doc5'}  # Accident-related documents
            },
            {
                'query': 'clear weather',
                'relevant_doc_ids': {'doc2', 'doc4'}  # Clear weather documents
            }
        ]
        
        # Create qrels dictionary
        for query_data in sample_queries:
            query_id = f"q_{len(sample_qrels) + 1}"
            sample_qrels[query_id] = query_data['relevant_doc_ids']
        
        print(f"Created {len(sample_qrels)} sample query relevance judgments")
        return sample_qrels
    
    def evaluate_system(self, 
                   queries: List[str], 
                   qrels: Dict[str, Set[str]],
                   k_values: List[int] = [1, 3, 5, 10],
                   output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the complete IR system
        
        Args:
            queries: List of queries to evaluate
            qrels: Relevance judgments for queries
            k_values: List of k values for Precision@K and Recall@K
            output_file: Optional file to save results
            
        Returns:
            Dictionary with evaluation results
        """
        print("=" * 80)
        print("ROUTIQ IR - SYSTEM EVALUATION")
        print("=" * 80)
        
        if not queries:
            print("No queries provided for evaluation")
            return {'message': 'No queries provided'}
        
        print(f"Evaluating {len(queries)} queries")
        print(f"Using k values: {k_values}")
        
        # Evaluate each query
        all_query_results = []
        
        for i, query_obj in enumerate(queries):
            # Extract query text from query object
            if isinstance(query_obj, dict):
                query_text = query_obj.get('text', str(query_obj))
                query_id = query_obj.get('query_id', f'q{i+1}')
            else:
                query_text = str(query_obj)
                query_id = f'q{i+1}'
            
            print(f"\nEvaluating query {i+1}: '{query_text}'")
            
            # Get search results
            search_results = self.retrieval_engine.search(query_text, strategy="smart", k=max(k_values))
            
            # Get relevant documents for this query
            if isinstance(query_obj, dict):
                relevant_docs_list = []
                for qrel in qrels:
                    if qrel.get('query_id') == query_id:
                        relevant_docs_list.extend(qrel.get('relevant_docs', []))
                relevant_docs = set(relevant_docs_list)
            else:
                relevant_docs = qrels.get(query, set())
            
            # Evaluate this query
            query_evaluation = self.metrics_calculator.evaluate_query(
                search_results['results'],
                relevant_docs,
                len(search_results['results']),  # Total docs in corpus
                len(relevant_docs),  # Total relevant docs
                k_values
            )
            
            # Add query info
            query_evaluation['query'] = query
            query_evaluation['query_number'] = i + 1
            
            all_query_results.append(query_evaluation)
        
        # Calculate final metrics
        final_metrics = self.metrics_calculator.calculate_final_metrics()
        
        # Prepare results
        evaluation_results = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_queries': len(queries),
                'k_values': k_values,
                'system_info': {
                    'indices_available': list(self.index_manager.list_indices().keys()),
                    'retrieval_engine': 'RetrievalEngine with smart routing'
                }
            },
            'query_results': all_query_results,
            'final_metrics': final_metrics,
            'detailed_results': {
                'precision_at_k': {k: self.metrics_calculator.precision_at_k[k] for k in k_values},
                'recall_at_k': {k: self.metrics_calculator.recall_at_k[k] for k in k_values},
                'ndcg_at_k': {k: self.metrics_calculator.ndcg_scores[k] for k in k_values}
            }
        }
        
        self.evaluation_results = evaluation_results
        
        # Print summary
        self._print_evaluation_summary(evaluation_results)
        
        # Save results if requested
        if output_file:
            self._save_evaluation_results(evaluation_results, output_file)
        
        return evaluation_results
    
    def _print_evaluation_summary(self, results: Dict[str, Any]):
        """Print formatted evaluation summary"""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        metadata = results['evaluation_metadata']
        final_metrics = results['final_metrics']
        
        print(f"Evaluation Date: {metadata['timestamp']}")
        print(f"Total Queries: {metadata['total_queries']}")
        print(f"K Values: {metadata['k_values']}")
        print(f"Available Indices: {metadata['system_info']['indices_available']}")
        
        print(f"\nOVERALL METRICS:")
        print(f"MAP Score: {final_metrics['map_score']:.4f}")
        print(f"Average Precision: {final_metrics['average_precision']:.4f}")
        
        print(f"\nPRECISION@K:")
        for k in metadata['k_values']:
            if f'avg_precision_at_{k}' in final_metrics:
                print(f"  @{k}: {final_metrics[f'avg_precision_at_{k}']:.4f}")
        
        print(f"\nRECALL@K:")
        for k in metadata['k_values']:
            if f'avg_recall_at_{k}' in final_metrics:
                print(f"  @{k}: {final_metrics[f'avg_recall_at_{k}']:.4f}")
        
        print(f"\nNDCG@K:")
        for k in metadata['k_values']:
            if f'avg_ndcg_at_{k}' in final_metrics:
                print(f"  @{k}: {final_metrics[f'avg_ndcg_at_{k}']:.4f}")
        
        print("\n" + "=" * 80)
    
    def _save_evaluation_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"Evaluation results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def create_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Create human-readable evaluation report"""
        report = []
        report.append("# RoutiQ IR System Evaluation Report")
        report.append(f"Generated: {results['evaluation_metadata']['timestamp']}")
        report.append("")
        
        metadata = results['evaluation_metadata']
        final_metrics = results['final_metrics']
        
        # System information
        report.append("## System Configuration")
        report.append(f"- Total Queries Evaluated: {metadata['total_queries']}")
        report.append(f"- K Values: {metadata['k_values']}")
        report.append(f"- Available Indices: {', '.join(metadata['system_info']['indices_available'])}")
        report.append("")
        
        # Overall metrics
        report.append("## Overall Performance")
        report.append(f"- MAP Score: {final_metrics['map_score']:.4f}")
        report.append(f"- Average Precision: {final_metrics['average_precision']:.4f}")
        report.append("")
        
        # Detailed metrics
        report.append("## Precision@K Results")
        for k in metadata['k_values']:
            if f'avg_precision_at_{k}' in final_metrics:
                report.append(f"- Precision@{k}: {final_metrics[f'avg_precision_at_{k}']:.4f}")
        
        report.append("")
        report.append("## Recall@K Results")
        for k in metadata['k_values']:
            if f'avg_recall_at_{k}' in final_metrics:
                report.append(f"- Recall@{k}: {final_metrics[f'avg_recall_at_{k}']:.4f}")
        
        report.append("")
        report.append("## NDCG@K Results")
        for k in metadata['k_values']:
            if f'avg_ndcg_at_{k}' in final_metrics:
                report.append(f"- NDCG@{k}: {final_metrics[f'avg_ndcg_at_{k}']:.4f}")
        
        return '\n'.join(report)
    
    def run_interactive_evaluation(self):
        """Run interactive evaluation with user input"""
        print("=" * 80)
        print("INTERACTIVE EVALUATION")
        print("=" * 80)
        
        # Get available indices
        available_indices = self.index_manager.list_indices()
        print(f"Available indices: {available_indices}")
        
        # Get evaluation queries
        queries_input = input("Enter queries (comma-separated, or 'sample' for sample queries): ").strip()
        
        if queries_input.lower() == 'sample':
            queries = [
                'heavy congestion',
                'rain weather', 
                'highway traffic',
                'rush hour',
                'accident',
                'clear weather'
            ]
        else:
            queries = [q.strip() for q in queries_input.split(',') if q.strip()]
        
        if not queries:
            print("No queries provided")
            return
        
        # Get k values
        k_input = input("Enter k values (comma-separated, default '1,3,5,10'): ").strip()
        if k_input:
            try:
                k_values = [int(k.strip()) for k in k_input.split(',') if k.strip()]
            except ValueError:
                k_values = [1, 3, 5, 10]
        else:
            k_values = [1, 3, 5, 10]
        
        # Get output file
        output_file = input("Enter output file (or press Enter for console only): ").strip()
        if not output_file:
            output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Create sample qrels if needed
        print("Create sample qrels? (y/n): ")
        create_qrels = input().strip().lower() == 'y'
        
        if create_qrels:
            print("Loading corpus for sample qrels...")
            # Load a sample of corpus for qrels creation
            sample_corpus_file = "data/processed/traffic_corpus_20260424_143258.pkl"
            try:
                import pickle
                with open(sample_corpus_file, 'rb') as f:
                    sample_corpus = pickle.load(f)[:100]  # First 100 docs
                sample_qrels = self.create_sample_qrels(sample_corpus)
            except Exception as e:
                print(f"Error loading corpus: {e}")
                sample_qrels = {}
        else:
            sample_qrels = {}
        
        # Run evaluation
        results = self.evaluate_system(queries, sample_qrels, k_values, output_file)
        
        print(f"\nEvaluation complete! Results saved to: {output_file}")
        return results
