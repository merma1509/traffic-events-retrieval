"""Index management for traffic events retrieval system"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from .bm25_indexer import BM25Indexer


class IndexManager:
    """Manage multiple indices and provide unified search interface"""
    
    def __init__(self, indices_dir: str = "data/indices"):
        self.indices_dir = Path(indices_dir)
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.indices = {}
        self.index_metadata = {}
        
    def create_bm25_index(self, 
                          corpus: List[Dict[str, Any]], 
                          index_name: str = "default",
                          token_field: str = 'all_tokens',
                          k1: float = 1.2,
                          b: float = 0.75) -> Dict[str, Any]:
        """Create and save a BM25 index"""
        print(f"Creating BM25 index: {index_name}")
        
        # Create indexer
        indexer = BM25Indexer(k1=k1, b=b)
        
        # Build index
        stats = indexer.build_index(corpus, token_field=token_field)
        
        # Save index
        index_path = self.indices_dir / f"{index_name}_bm25.pkl"
        indexer.save_index(str(index_path))
        
        # Save metadata
        metadata = {
            'index_name': index_name,
            'index_type': 'bm25',
            'token_field': token_field,
            'parameters': {'k1': k1, 'b': b},
            'stats': stats,
            'created_at': datetime.now().isoformat(),
            'corpus_size': len(corpus),
            'index_path': str(index_path)
        }
        
        metadata_path = self.indices_dir / f"{index_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Store in memory
        self.indices[index_name] = indexer
        self.index_metadata[index_name] = metadata
        
        print(f"BM25 index '{index_name}' created successfully")
        return metadata
    
    def load_index(self, index_name: str) -> bool:
        """Load an existing index"""
        index_path = self.indices_dir / f"{index_name}_bm25.pkl"
        metadata_path = self.indices_dir / f"{index_name}_metadata.json"
        
        if not index_path.exists() or not metadata_path.exists():
            print(f"Index '{index_name}' not found")
            return False
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load index
            indexer = BM25Indexer()
            if indexer.load_index(str(index_path)):
                self.indices[index_name] = indexer
                self.index_metadata[index_name] = metadata
                print(f"Index '{index_name}' loaded successfully")
                return True
            else:
                print(f"Failed to load index '{index_name}'")
                return False
                
        except Exception as e:
            print(f"Error loading index '{index_name}': {e}")
            return False
    
    def search(self, 
               query: str, 
               index_name: str = "default",
               k: int = 10) -> List[Dict[str, Any]]:
        """Search using specified index"""
        if index_name not in self.indices:
            if not self.load_index(index_name):
                return []
        
        indexer = self.indices[index_name]
        results = indexer.search(query, k=k)
        
        # Format results
        formatted_results = []
        for doc_id, score, doc in results:
            formatted_results.append({
                'doc_id': doc.get('doc_id', f'doc_{doc_id}'),
                'score': score,
                'document': doc,
                'index_name': index_name
            })
        
        return formatted_results
    
    def multi_index_search(self, 
                          query: str, 
                          indices: List[str] = None,
                          k: int = 10) -> List[Dict[str, Any]]:
        """Search across multiple indices and merge results"""
        if indices is None:
            indices = list(self.indices.keys())
        
        all_results = []
        
        for index_name in indices:
            if index_name in self.indices:
                results = self.search(query, index_name, k * 2)  # Get more results per index
                all_results.extend(results)
        
        # Sort by score and return top k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:k]
    
    def get_index_stats(self, index_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for an index"""
        if index_name not in self.index_metadata:
            return None
        
        return self.index_metadata[index_name]
    
    def list_indices(self) -> List[str]:
        """List all available indices"""
        # First, try to load any existing indices on disk
        self._discover_and_load_indices()
        return list(self.indices.keys())
    
    def _discover_and_load_indices(self):
        """Discover and load existing indices from disk"""
        if not self.indices_dir.exists():
            return
        
        # Look for metadata files
        for metadata_file in self.indices_dir.glob("*_metadata.json"):
            index_name = metadata_file.stem.replace("_metadata", "")
            if index_name not in self.indices:
                self.load_index(index_name)
    
    def create_specialized_indices(self, corpus: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create specialized indices for different token types"""
        print("Creating specialized indices...")
        
        indices_created = {}
        
        # Main index (all tokens)
        main_metadata = self.create_bm25_index(
            corpus, index_name="main", token_field="all_tokens"
        )
        indices_created["main"] = main_metadata
        
        # Congestion-focused index
        congestion_metadata = self.create_bm25_index(
            corpus, index_name="congestion", token_field="congestion_tokens"
        )
        indices_created["congestion"] = congestion_metadata
        
        # Weather-focused index
        weather_metadata = self.create_bm25_index(
            corpus, index_name="weather", token_field="weather_tokens"
        )
        indices_created["weather"] = weather_metadata
        
        # Spatial-focused index
        spatial_metadata = self.create_bm25_index(
            corpus, index_name="spatial", token_field="spatial_tokens"
        )
        indices_created["spatial"] = spatial_metadata
        
        # Temporal-focused index
        temporal_metadata = self.create_bm25_index(
            corpus, index_name="temporal", token_field="temporal_tokens"
        )
        indices_created["temporal"] = temporal_metadata
        
        print(f"Created {len(indices_created)} specialized indices")
        return indices_created
    
    def smart_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Smart search that determines the best index to use"""
        # Simple heuristic: check query terms and route to appropriate index
        query_lower = query.lower()
        
        congestion_terms = ['congestion', 'traffic', 'jam', 'delays', 'heavy', 'moderate', 'light']
        weather_terms = ['rain', 'weather', 'storm', 'clear', 'sunny', 'fog', 'snow']
        spatial_terms = ['road', 'highway', 'street', 'motorway', 'node', 'route']
        temporal_terms = ['morning', 'evening', 'rush', 'hour', 'day', 'night']
        
        # Determine primary focus
        congestion_score = sum(1 for term in congestion_terms if term in query_lower)
        weather_score = sum(1 for term in weather_terms if term in query_lower)
        spatial_score = sum(1 for term in spatial_terms if term in query_lower)
        temporal_score = sum(1 for term in temporal_terms if term in query_lower)
        
        scores = {
            'congestion': congestion_score,
            'weather': weather_score,
            'spatial': spatial_score,
            'temporal': temporal_score
        }
        
        # Select best index
        best_index = max(scores, key=scores.get)
        if scores[best_index] == 0:
            best_index = 'main'  # Default to main index
        
        print(f"Smart search: using '{best_index}' index (scores: {scores})")
        
        return self.search(query, best_index, k)
    
    def export_index_stats(self, output_file: str = None) -> str:
        """Export all index statistics to JSON"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.indices_dir / f"index_stats_{timestamp}.json"
        
        stats = {
            'export_time': datetime.now().isoformat(),
            'total_indices': len(self.indices),
            'indices': {}
        }
        
        for index_name, metadata in self.index_metadata.items():
            stats['indices'][index_name] = {
                'index_type': metadata['index_type'],
                'corpus_size': metadata['corpus_size'],
                'vocab_size': metadata['stats']['vocab_size'],
                'avg_doc_length': metadata['stats']['avg_doc_length'],
                'build_time': metadata['stats']['build_time'],
                'created_at': metadata['created_at']
            }
        
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Index statistics exported to {output_file}")
        return str(output_file)
    
    def print_summary(self):
        """Print summary of all indices"""
        print("\n" + "=" * 80)
        print("INDEX MANAGER SUMMARY")
        print("=" * 80)
        
        if not self.indices:
            print("No indices loaded")
            return
        
        for index_name, metadata in self.index_metadata.items():
            print(f"\nIndex: {index_name}")
            print(f"  Type: {metadata['index_type']}")
            print(f"  Token Field: {metadata['token_field']}")
            print(f"  Corpus Size: {metadata['corpus_size']:,}")
            print(f"  Vocabulary: {metadata['stats']['vocab_size']:,}")
            print(f"  Avg Doc Length: {metadata['stats']['avg_doc_length']:.1f}")
            print(f"  Build Time: {metadata['stats']['build_time']:.1f}s")
            print(f"  Created: {metadata['created_at']}")
