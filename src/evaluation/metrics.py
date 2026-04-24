"""Evaluation metrics for traffic events retrieval system"""

import math
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict


class EvaluationMetrics:
    """Information retrieval evaluation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.queries_evaluated = 0
        self.retrieved_docs = []
        self.relevant_docs = []
        self.scores = []
        self.precision_at_k = defaultdict(list)
        self.recall_at_k = defaultdict(list)
        self.map_score = 0.0
        self.ndcg_scores = defaultdict(list)
    
    def calculate_precision_at_k(self, retrieved: List[Dict[str, Any]], relevant: Set[str], k: int) -> float:
        """Calculate Precision@K"""
        if not retrieved or k <= 0:
            return 0.0
        
        # Count relevant documents in top k
        relevant_in_top_k = sum(1 for doc in retrieved[:k] if doc['doc_id'] in relevant)
        
        return relevant_in_top_k / min(k, len(retrieved))
    
    def calculate_recall_at_k(self, retrieved: List[Dict[str, Any]], relevant: Set[str], total_relevant: int, k: int) -> float:
        """Calculate Recall@K"""
        if not retrieved or k <= 0 or total_relevant == 0:
            return 0.0
        
        # Count relevant documents in top k
        relevant_in_top_k = sum(1 for doc in retrieved[:k] if doc['doc_id'] in relevant)
        
        return relevant_in_top_k / min(total_relevant, k)
    
    def calculate_average_precision(self, retrieved: List[Dict[str, Any]], relevant: Set[str]) -> float:
        """Calculate average precision"""
        if not retrieved:
            return 0.0
        
        relevant_retrieved = sum(1 for doc in retrieved if doc['doc_id'] in relevant)
        return relevant_retrieved / len(retrieved)
    
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score"""
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_dcg(self, retrieved: List[Dict[str, Any]], relevant: Set[str], k: int) -> float:
        """Calculate Discounted Cumulative Gain (DCG)"""
        dcg = 0.0
        for i, doc in enumerate(retrieved[:k]):
            if doc['doc_id'] in relevant:
                # Relevance score (binary: 1 for relevant, 0 for non-relevant)
                relevance = 1.0
            else:
                relevance = 0.0
            
            # Discount factor: 1/log2(i+1)
            discount = 1.0 / math.log2(i + 2)
            dcg += relevance * discount
        
        return dcg
    
    def calculate_idcg(self, relevant: Set[str], k: int) -> float:
        """Calculate Ideal DCG"""
        # Sort relevant documents by relevance (all have relevance = 1)
        sorted_relevant = sorted(list(relevant), key=lambda x: 1)  # All have same relevance
        
        dcg = 0.0
        for i in range(min(k, len(sorted_relevant))):
            # Discount factor
            discount = 1.0 / math.log2(i + 2)
            dcg += discount
        
        return dcg
    
    def calculate_ndcg_at_k(self, retrieved: List[Dict[str, Any]], relevant: Set[str], k: int) -> float:
        """Calculate Normalized DCG (NDCG)@K"""
        dcg = self.calculate_dcg(retrieved, relevant, k)
        idcg = self.calculate_idcg(relevant, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_map(self, queries_results: List[Dict[str, Any]]) -> float:
        """Calculate Mean Average Precision (MAP)"""
        if not queries_results:
            return 0.0
        
        average_precisions = []
        
        for query_result in queries_results:
            retrieved = query_result['retrieved']
            relevant = query_result['relevant']
            
            if retrieved and relevant:
                # Calculate average precision for this query
                precisions = []
                for k in range(1, len(retrieved) + 1):
                    precision_k = self.calculate_precision_at_k(retrieved, relevant, k)
                    precisions.append(precision_k)
                
                # Average precision (AP) for this query
                avg_precision = sum(precisions) / len(precisions)
                average_precisions.append(avg_precision)
        
        return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    
    def evaluate_query(self, 
                    query: str,
                    retrieved: List[Dict[str, Any]], 
                    relevant: Set[str],
                    k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Evaluate a single query
        
        Args:
            query: The search query
            retrieved: List of retrieved documents with scores
            relevant: Set of relevant document IDs
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        evaluation = {
            'query': query,
            'retrieved_count': len(retrieved),
            'relevant_count': len(relevant),
            'relevant_retrieved': len([doc for doc in retrieved if doc['doc_id'] in relevant])
        }
        
        # Calculate Precision@K for different k values
        for k in k_values:
            evaluation[f'precision_at_{k}'] = self.calculate_precision_at_k(retrieved, relevant, k)
            evaluation[f'recall_at_{k}'] = self.calculate_recall_at_k(retrieved, relevant, len(relevant), k)
        
        # Calculate other metrics
        evaluation['average_precision'] = self.calculate_average_precision(retrieved, relevant)
        evaluation['f1_score'] = self.calculate_f1_score(
            evaluation['average_precision'],
            evaluation.get(f'recall_at_{k_values[-1]}', 0.0)  # Use highest k
        )
        
        # Calculate DCG and NDCG
        for k in k_values:
            evaluation[f'dcg_at_{k}'] = self.calculate_dcg(retrieved, relevant, k)
            evaluation[f'ndcg_at_{k}'] = self.calculate_ndcg_at_k(retrieved, relevant, k)
        
        return evaluation
    
    def update_running_metrics(self, query_evaluation: Dict[str, Any]):
        """Update running metrics for cumulative evaluation"""
        self.queries_evaluated += 1
        self.retrieved_docs.extend(query_evaluation.get('retrieved', []))
        self.relevant_docs.extend(query_evaluation.get('relevant', []))
        self.scores.append(query_evaluation)
        
        # Update Precision@K and Recall@K running averages
        for k in [1, 3, 5, 10]:
            if f'precision_at_{k}' in query_evaluation:
                self.precision_at_k[k].append(query_evaluation[f'precision_at_{k}'])
                self.recall_at_k[k].append(query_evaluation[f'recall_at_{k}'])
    
    def calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculate final evaluation metrics"""
        if self.queries_evaluated == 0:
            return {'message': 'No queries evaluated'}
        
        # Calculate MAP
        self.map_score = self.calculate_map(self.scores)
        
        # Calculate average Precision@K and Recall@K
        final_metrics = {
            'total_queries': self.queries_evaluated,
            'map_score': self.map_score,
            'average_precision': sum(self.scores) / len(self.scores) if self.scores else 0.0
        }
        
        # Calculate average Precision@K and Recall@K
        for k in [1, 3, 5, 10]:
            if self.precision_at_k[k]:
                final_metrics[f'avg_precision_at_{k}'] = sum(self.precision_at_k[k]) / len(self.precision_at_k[k])
                final_metrics[f'avg_recall_at_{k}'] = sum(self.recall_at_k[k]) / len(self.recall_at_k[k])
        
        # Calculate NDCG averages
        for k in [1, 3, 5, 10]:
            if self.ndcg_scores[k]:
                final_metrics[f'avg_ndcg_at_{k}'] = sum(self.ndcg_scores[k]) / len(self.ndcg_scores[k])
        
        return final_metrics
    
    def print_evaluation_summary(self, metrics: Dict[str, Any]):
        """Print formatted evaluation summary"""
        print("\n" + "=" * 80)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 80)
        
        if 'message' in metrics:
            print(metrics['message'])
            return
        
        print(f"Total Queries: {metrics['total_queries']}")
        print(f"MAP Score: {metrics['map_score']:.4f}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")
        
        # Print Precision@K and Recall@K
        print(f"\nPrecision@K:")
        for k in [1, 3, 5, 10]:
            if f'avg_precision_at_{k}' in metrics:
                print(f"  @{k}: {metrics[f'avg_precision_at_{k}']:.4f}")
        
        print(f"\nRecall@K:")
        for k in [1, 3, 5, 10]:
            if f'avg_recall_at_{k}' in metrics:
                print(f"  @{k}: {metrics[f'avg_recall_at_{k}']:.4f}")
        
        # Print NDCG
        print(f"\nNDCG@K:")
        for k in [1, 3, 5, 10]:
            if f'avg_ndcg_at_{k}' in metrics:
                print(f"  @{k}: {metrics[f'avg_ndcg_at_{k}']:.4f}")
        
        print("\n" + "=" * 80)
