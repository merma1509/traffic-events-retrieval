"""Retrieval engine for traffic events search system"""

import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .query_processor import QueryProcessor
from indexing import IndexManager


class RetrievalEngine:
    """Retrieval engine with multiple search strategies"""
    
    def __init__(self, indices_dir: str = "data/indices"):
        self.query_processor = QueryProcessor()
        self.index_manager = IndexManager(indices_dir)
        self.search_stats = {
            'total_searches': 0,
            'search_times': [],
            'result_counts': []
        }
    
    def search(self, 
               query: str, 
               strategy: str = "smart",
               k: int = 10,
               index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform search with specified strategy
        
        Args:
            query: Search query string
            strategy: Search strategy ('smart', 'basic', 'multi', 'specialized')
            k: Number of results to return
            index_name: Specific index to use (overrides strategy)
            
        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()
        
        # Process query
        query_data = self.query_processor.preprocess_query(query)
        
        # Choose search strategy
        if index_name:
            results = self.index_manager.search(query, index_name, k)
            strategy_used = f"specific_index_{index_name}"
        elif strategy == "smart":
            results = self.index_manager.smart_search(query, k)
            strategy_used = "smart_routing"
        elif strategy == "multi":
            results = self.index_manager.multi_index_search(query, k=k)
            strategy_used = "multi_index"
        elif strategy == "basic":
            results = self.index_manager.search(query, "main", k)
            strategy_used = "basic_main"
        else:
            # Default to specialized based on intent
            intent = query_data['intent_analysis']['primary_intent']
            if intent in ['congestion', 'weather', 'spatial', 'temporal']:
                results = self.index_manager.search(query, intent, k)
                strategy_used = f"specialized_{intent}"
            else:
                results = self.index_manager.search(query, "main", k)
                strategy_used = "fallback_main"
        
        search_time = time.time() - start_time
        
        # Update statistics
        self.search_stats['total_searches'] += 1
        self.search_stats['search_times'].append(search_time)
        self.search_stats['result_counts'].append(len(results))
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'doc_id': result['doc_id'],
                'score': result['score'],
                'document': result['document'],
                'relevance_features': self._extract_relevance_features(result['document'], query_data),
                'snippet': self._generate_snippet(result['document'], query_data['tokens'])
            })
        
        return {
            'query': query,
            'query_data': query_data,
            'results': formatted_results,
            'metadata': {
                'strategy_used': strategy_used,
                'search_time': search_time,
                'total_results': len(results),
                'returned_results': len(formatted_results),
                'index_used': result['index_name'] if results else None
            }
        }
    
    def advanced_search(self, 
                       query: str,
                       filters: Optional[Dict[str, Any]] = None,
                       ranking: str = "bm25",
                       k: int = 10) -> Dict[str, Any]:
        """
        Advanced search with filtering and ranking options
        
        Args:
            query: Search query
            filters: Dictionary of filters (e.g., {'congestion_level': 'Heavy', 'time_range': 'morning'})
            ranking: Ranking method ('bm25', 'tfidf', 'recent')
            k: Number of results
            
        Returns:
            Enhanced search results with filtering
        """
        # Perform basic search first
        search_results = self.search(query, strategy="smart", k=k*2)  # Get more for filtering
        
        # Apply filters
        if filters:
            filtered_results = self._apply_filters(search_results['results'], filters)
        else:
            filtered_results = search_results['results']
        
        # Apply ranking
        if ranking == "recent":
            # Sort by timestamp (most recent first)
            filtered_results.sort(key=lambda x: x['document'].get('timestamp', ''), reverse=True)
        elif ranking == "tfidf":
            # TF-IDF ranking (simplified - would need precomputed TF-IDF)
            filtered_results.sort(key=lambda x: len(x['document'].get('all_tokens', [])), reverse=True)
        # BM25 is default (already sorted)
        
        # Limit results
        final_results = filtered_results[:k]
        
        return {
            'query': query,
            'results': final_results,
            'metadata': {
                'filters_applied': filters or {},
                'ranking_method': ranking,
                'original_results': len(search_results['results']),
                'filtered_results': len(filtered_results),
                'final_results': len(final_results),
                'search_time': search_results['metadata']['search_time']
            }
        }
    
    def _extract_relevance_features(self, document: Dict[str, Any], query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevance features for result ranking and display"""
        features = {}
        
        # Token overlap
        doc_tokens = set(document.get('all_tokens', []))
        query_tokens = set(query_data['expanded_terms'])
        
        overlap = doc_tokens & query_tokens
        features['token_overlap'] = len(overlap)
        features['token_overlap_ratio'] = len(overlap) / len(query_tokens) if query_tokens else 0
        features['doc_token_coverage'] = len(overlap) / len(doc_tokens) if doc_tokens else 0
        
        # Field-specific matches
        features['congestion_match'] = bool(overlap & set(document.get('congestion_tokens', [])))
        features['weather_match'] = bool(overlap & set(document.get('weather_tokens', [])))
        features['spatial_match'] = bool(overlap & set(document.get('spatial_tokens', [])))
        features['temporal_match'] = bool(overlap & set(document.get('temporal_tokens', [])))
        
        # Document quality indicators
        features['has_coordinates'] = bool(document.get('source_x') and document.get('source_y'))
        features['has_weather_data'] = bool(document.get('weather_condition'))
        features['has_time_data'] = bool(document.get('timestamp'))
        features['token_count'] = document.get('token_count', 0)
        
        return features
    
    def _generate_snippet(self, document: Dict[str, Any], query_tokens: List[str]) -> str:
        """Generate search snippet with highlighted terms"""
        text = document.get('text', '')
        
        if not text or not query_tokens:
            return text[:200] + "..." if len(text) > 200 else text
        
        # Simple snippet generation - find first query term occurrence
        for term in query_tokens:
            if term.lower() in text.lower():
                start_idx = text.lower().find(term.lower())
                if start_idx != -1:
                    # Extract context around the term
                    context_start = max(0, start_idx - 50)
                    context_end = min(len(text), start_idx + len(term) + 50)
                    snippet = text[context_start:context_end]
                    
                    # Highlight the term (simple approach)
                    highlighted = snippet.replace(term, f"**{term}**", 1)
                    return highlighted
        
        # Fallback to first 200 characters
        return text[:200] + "..." if len(text) > 200 else text
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to search results"""
        filtered_results = []
        
        for result in results:
            document = result['document']
            include_result = True
            
            # Apply each filter
            for filter_key, filter_value in filters.items():
                if filter_key == 'congestion_level':
                    if document.get('congestion_level') != filter_value:
                        include_result = False
                elif filter_key == 'time_range':
                    # Filter by time of day
                    hour = document.get('hour_of_day')
                    if filter_value == 'morning' and not (6 <= hour <= 10):
                        include_result = False
                    elif filter_value == 'evening' and not (17 <= hour <= 20):
                        include_result = False
                    elif filter_value == 'rush_hour' and not document.get('is_rush_hour'):
                        include_result = False
                elif filter_key == 'weather_condition':
                    if document.get('weather_condition') != filter_value:
                        include_result = False
                elif filter_key == 'has_coordinates':
                    if filter_value and not (document.get('source_x') and document.get('source_y')):
                        include_result = False
                elif filter_key == 'min_token_count':
                    if document.get('token_count', 0) < filter_value:
                        include_result = False
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results
    
    def get_search_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on query and common patterns"""
        return self.query_processor.get_query_suggestions(query, limit)
    
    def explain_search(self, query: str, strategy: str = "smart") -> str:
        """Generate explanation of search processing"""
        explanation = []
        explanation.append(f"Search Query: '{query}'")
        explanation.append(f"Strategy: {strategy}")
        
        # Process query
        query_data = self.query_processor.preprocess_query(query)
        explanation.append(self.query_processor.explain_query_processing(query_data))
        
        return '\n'.join(explanation)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        if not self.search_stats['search_times']:
            return {'message': 'No searches performed yet'}
        
        return {
            'total_searches': self.search_stats['total_searches'],
            'avg_search_time': sum(self.search_stats['search_times']) / len(self.search_stats['search_times']),
            'min_search_time': min(self.search_stats['search_times']),
            'max_search_time': max(self.search_stats['search_times']),
            'avg_results_per_search': sum(self.search_stats['result_counts']) / len(self.search_stats['result_counts']),
            'total_results_returned': sum(self.search_stats['result_counts'])
        }
    
    def batch_search(self, queries: List[str], strategy: str = "smart", k: int = 10) -> List[Dict[str, Any]]:
        """Perform multiple searches efficiently"""
        batch_results = []
        
        for query in queries:
            result = self.search(query, strategy=strategy, k=k)
            batch_results.append(result)
        
        return batch_results
    
    def export_results(self, results: Dict[str, Any], format: str = "json", filename: Optional[str] = None) -> str:
        """Export search results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"search_results_{timestamp}.{format}"
        
        if format == "json":
            import json
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        elif format == "csv":
            import pandas as pd
            
            # Flatten results for CSV
            flattened = []
            for result in results.get('results', []):
                flat_result = {
                    'query': results.get('query'),
                    'doc_id': result['doc_id'],
                    'score': result['score'],
                    'text': result['document'].get('text', ''),
                    'congestion_level': result['document'].get('congestion_level', ''),
                    'weather_condition': result['document'].get('weather_condition', ''),
                    'timestamp': result['document'].get('timestamp', ''),
                    'snippet': result.get('snippet', '')
                }
                flattened.append(flat_result)
            
            df = pd.DataFrame(flattened)
            df.to_csv(filename, index=False, encoding='utf-8')
        
        print(f"Results exported to {filename}")
        return filename
