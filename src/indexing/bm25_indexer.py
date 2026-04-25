"""BM25 indexing components for traffic events retrieval system"""

import math
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
from datetime import datetime


class BM25Indexer:
    """BM25 indexing and retrieval for traffic events"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 indexer
        
        Args:
            k1: Controls term frequency saturation
            b: Controls document length normalization
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_length = 0.0
        self.doc_lengths = []
        self.vocab = set()
        self.term_doc_freq = defaultdict(int)  # Document frequency
        self.term_index = defaultdict(list)    # Inverted index: term -> [(doc_id, tf), ...]
        self.doc_store = {}  # Store original documents
        self.index_stats = {}
        
    def build_index(self, corpus: List[Dict[str, Any]], token_field: str = 'all_tokens') -> Dict[str, Any]:
        """
        Build BM25 index from processed corpus
        
        Args:
            corpus: List of processed documents
            token_field: Field name containing tokens
        """
        print(f"Building BM25 index from {len(corpus):,} documents...")
        start_time = datetime.now()
        
        self.corpus_size = len(corpus)
        self.doc_lengths = []
        all_tokens = []
        
        # Build inverted index
        for doc_id, doc in enumerate(corpus):
            tokens = doc.get(token_field, [])
            if not tokens:
                continue
                
            # Store original document
            self.doc_store[doc_id] = doc
            
            # Calculate document length
            doc_length = len(tokens)
            self.doc_lengths.append(doc_length)
            all_tokens.extend(tokens)
            
            # Count term frequencies in this document
            term_counts = Counter(tokens)
            
            # Add to inverted index
            for term, tf in term_counts.items():
                self.term_index[term].append((doc_id, tf))
                self.vocab.add(term)
        
        # Calculate statistics
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # Calculate document frequencies
        for term, postings in self.term_index.items():
            self.term_doc_freq[term] = len(postings)
        
        # Build index statistics
        build_time = (datetime.now() - start_time).total_seconds()
        self.index_stats = {
            'build_time': build_time,
            'corpus_size': self.corpus_size,
            'vocab_size': len(self.vocab),
            'avg_doc_length': self.avg_doc_length,
            'total_tokens': len(all_tokens),
            'unique_terms': len(self.term_index),
            'postings_count': sum(len(postings) for postings in self.term_index.values()),
            'index_size_mb': self._estimate_index_size()
        }
        
        print(f"Index built in {build_time:.1f} seconds")
        print(f"Vocabulary: {len(self.vocab):,} unique terms")
        print(f"Postings: {self.index_stats['postings_count']:,}")
        print(f"Average document length: {self.avg_doc_length:.1f} tokens")
        
        return self.index_stats
    
    def _estimate_index_size(self) -> float:
        """Estimate index size in MB"""
        total_entries = sum(len(postings) for postings in self.term_index.values())
        # Rough estimate: each posting ~ (doc_id + tf) * 8 bytes
        return (total_entries * 16) / (1024 * 1024)
    
    def score_document(self, doc_id: int, query_terms: List[str]) -> float:
        """
        Calculate BM25 score for a document given query terms
        
        Args:
            doc_id: Document ID
            query_terms: List of query terms
            
        Returns:
            BM25 score
        """
        if doc_id not in self.doc_store:
            return 0.0
            
        doc_length = self.doc_lengths[doc_id]
        score = 0.0
        
        for term in query_terms:
            if term not in self.term_index:
                continue
                
            # Get term frequency in document
            tf = 0
            for posting_doc_id, posting_tf in self.term_index[term]:
                if posting_doc_id == doc_id:
                    tf = posting_tf
                    break
            
            if tf == 0:
                continue
            
            # Calculate BM25 components
            df = self.term_doc_freq[term]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5))
            
            # Normalized term frequency component
            normalized_tf = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            )
            
            score += idf * normalized_tf
        
        return score
    
    def search(self, query: str, k: int = 10, token_field: str = 'all_tokens') -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search documents using BM25
        
        Args:
            query: Query string
            k: Number of results to return
            token_field: Field to search in
            
        Returns:
            List of (doc_id, score, document) tuples
        """
        # Enhanced tokenization and query expansion
        query_terms = query.lower().split()
        query_terms = [term.strip() for term in query_terms if term.strip()]
        
        # Query expansion based on vocabulary
        expanded_terms = []
        for term in query_terms:
            # Direct match
            if term in self.term_index:
                expanded_terms.append(term)
            
            # Fuzzy matching for compound terms
            for vocab_term in self.term_index.keys():
                if term in vocab_term.split('_'):
                    expanded_terms.append(vocab_term)
        
        # Remove duplicates
        query_terms = list(set(expanded_terms))
        
        if not query_terms:
            return []
        
        # Find candidate documents (containing any query term)
        candidate_docs = set()
        for term in query_terms:
            if term in self.term_index:
                candidate_docs.update(doc_id for doc_id, _ in self.term_index[term])
        
        # Score all candidate documents
        scored_docs = []
        for doc_id in candidate_docs:
            score = self.score_document(doc_id, query_terms)
            if score > 0:
                scored_docs.append((doc_id, score, self.doc_store[doc_id]))
        
        # Sort by score (descending) and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]
    
    def get_term_statistics(self, term: str) -> Dict[str, Any]:
        """Get statistics for a specific term"""
        if term not in self.term_index:
            return {'term': term, 'df': 0, 'postings': []}
        
        postings = self.term_index[term]
        df = len(postings)
        
        # Calculate term frequency statistics
        tfs = [tf for _, tf in postings]
        
        return {
            'term': term,
            'df': df,
            'idf': math.log((self.corpus_size - df + 0.5) / (df + 0.5)),
            'total_tf': sum(tfs),
            'avg_tf': sum(tfs) / len(tfs),
            'max_tf': max(tfs),
            'postings': postings[:10]  # Return first 10 postings
        }
    
    def get_vocabulary_sample(self, n: int = 100) -> List[str]:
        """Get a sample of vocabulary terms"""
        vocab_list = list(self.vocab)
        return sorted(vocab_list)[:n]
    
    def save_index(self, filepath: str) -> str:
        """Save index to file"""
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'k1': self.k1,
            'b': self.b,
            'corpus_size': self.corpus_size,
            'avg_doc_length': self.avg_doc_length,
            'doc_lengths': self.doc_lengths,
            'vocab': list(self.vocab),
            'term_doc_freq': dict(self.term_doc_freq),
            'term_index': dict(self.term_index),
            'doc_store': self.doc_store,
            'index_stats': self.index_stats,
            'created_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        file_size = Path(filepath).stat().st_size / (1024 * 1024)  # MB
        print(f"Index saved to {filepath} ({file_size:.1f} MB)")
        return filepath
    
    def load_index(self, filepath: str) -> bool:
        """Load index from file"""
        try:
            with open(filepath, 'rb') as f:
                index_data = pickle.load(f)
            
            self.k1 = index_data['k1']
            self.b = index_data['b']
            self.corpus_size = index_data['corpus_size']
            self.avg_doc_length = index_data['avg_doc_length']
            self.doc_lengths = index_data['doc_lengths']
            self.vocab = set(index_data['vocab'])
            self.term_doc_freq = defaultdict(int, index_data['term_doc_freq'])
            self.term_index = defaultdict(list, index_data['term_index'])
            self.doc_store = index_data['doc_store']
            self.index_stats = index_data['index_stats']
            
            print(f"Index loaded from {filepath}")
            print(f"Corpus: {self.corpus_size:,} documents")
            print(f"Vocabulary: {len(self.vocab):,} terms")
            return True
            
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def print_index_summary(self):
        """Print index summary statistics"""
        print("\n" + "=" * 60)
        print("BM25 INDEX SUMMARY")
        print("=" * 60)
        
        if self.index_stats:
            stats = self.index_stats
            print(f"Documents: {stats['corpus_size']:,}")
            print(f"Vocabulary: {stats['vocab_size']:,} unique terms")
            print(f"Total tokens: {stats['total_tokens']:,}")
            print(f"Average doc length: {stats['avg_doc_length']:.1f} tokens")
            print(f"Postings: {stats['postings_count']:,}")
            print(f"Index size: {stats['index_size_mb']:.1f} MB")
            print(f"Build time: {stats['build_time']:.1f} seconds")
        else:
            print("No index built yet")
