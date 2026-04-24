"""Corpus saving components for traffic events retrieval system"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle
from datetime import datetime


class CorpusSaver:
    """Save processed traffic event corpus to various formats"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_to_json(self, corpus: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Save corpus to JSON format"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traffic_corpus_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare corpus for JSON serialization
        serializable_corpus = []
        for doc in corpus:
            serializable_doc = {}
            for key, value in doc.items():
                if isinstance(value, (str, int, float, bool, list)):
                    serializable_doc[key] = value
                else:
                    # Convert non-serializable objects to strings
                    serializable_doc[key] = str(value)
            serializable_corpus.append(serializable_doc)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_corpus, f, indent=2, ensure_ascii=False)
        
        print(f"Corpus saved to JSON: {filepath}")
        print(f"Documents: {len(corpus)}")
        return str(filepath)
    
    def save_to_csv(self, corpus: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Save corpus to CSV format"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traffic_corpus_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Flatten corpus for CSV
        flattened_corpus = []
        for doc in corpus:
            flat_doc = {}
            for key, value in doc.items():
                if isinstance(value, list):
                    # Convert lists to strings for CSV
                    flat_doc[key] = ';'.join(str(v) for v in value) if value else ''
                elif isinstance(value, (dict, set, tuple)):
                    # Convert complex objects to strings
                    flat_doc[key] = str(value)
                else:
                    flat_doc[key] = value
            flattened_corpus.append(flat_doc)
        
        df = pd.DataFrame(flattened_corpus)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"Corpus saved to CSV: {filepath}")
        print(f"Documents: {len(corpus)}, Columns: {len(df.columns)}")
        return str(filepath)
    
    def save_to_pickle(self, corpus: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Save corpus to pickle format (preserves Python objects)"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"traffic_corpus_{timestamp}.pkl"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(corpus, f)
        
        print(f"Corpus saved to Pickle: {filepath}")
        print(f"Documents: {len(corpus)}")
        return str(filepath)
    
    def save_corpus_summary(self, corpus: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Save corpus summary statistics"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"corpus_summary_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Calculate summary statistics
        summary = {
            "corpus_info": {
                "total_documents": len(corpus),
                "created_at": datetime.now().isoformat(),
                "document_fields": list(corpus[0].keys()) if corpus else []
            },
            "token_statistics": {},
            "congestion_distribution": {},
            "weather_distribution": {},
            "spatial_coverage": {},
            "temporal_coverage": {}
        }
        
        # Token statistics
        token_counts = [doc.get('token_count', 0) for doc in corpus]
        if token_counts:
            summary["token_statistics"] = {
                "total_tokens": sum(token_counts),
                "avg_tokens_per_doc": sum(token_counts) / len(token_counts),
                "min_tokens": min(token_counts),
                "max_tokens": max(token_counts),
                "unique_tokens_in_corpus": len(set(token for doc in corpus for token in doc.get('all_tokens', [])))
            }
        
        # Congestion distribution
        congestion_levels = [doc.get('congestion_level', 'Unknown') for doc in corpus]
        congestion_counts = {level: congestion_levels.count(level) for level in set(congestion_levels)}
        summary["congestion_distribution"] = congestion_counts
        
        # Weather distribution
        weather_conditions = [doc.get('weather_condition', 'Unknown') for doc in corpus]
        weather_counts = {condition: weather_conditions.count(condition) for condition in set(weather_conditions)}
        summary["weather_distribution"] = weather_counts
        
        # Spatial coverage
        spatial_tokens = sum(1 for doc in corpus if doc.get('spatial_tokens'))
        summary["spatial_coverage"] = {
            "documents_with_spatial_tokens": spatial_tokens,
            "percentage": (spatial_tokens / len(corpus)) * 100 if corpus else 0
        }
        
        # Temporal coverage
        temporal_tokens = sum(1 for doc in corpus if doc.get('temporal_tokens'))
        summary["temporal_coverage"] = {
            "documents_with_temporal_tokens": temporal_tokens,
            "percentage": (temporal_tokens / len(corpus)) * 100 if corpus else 0
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Corpus summary saved to: {filepath}")
        return str(filepath)
    
    def save_all_formats(self, corpus: List[Dict[str, Any]], base_filename: Optional[str] = None) -> Dict[str, str]:
        """Save corpus in all formats"""
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"traffic_corpus_{timestamp}"
        
        saved_files = {}
        
        # Save in all formats
        saved_files['json'] = self.save_to_json(corpus, f"{base_filename}.json")
        saved_files['csv'] = self.save_to_csv(corpus, f"{base_filename}.csv")
        saved_files['pickle'] = self.save_to_pickle(corpus, f"{base_filename}.pkl")
        saved_files['summary'] = self.save_corpus_summary(corpus, f"{base_filename}_summary.json")
        
        print(f"\nAll corpus files saved:")
        for format_type, filepath in saved_files.items():
            print(f"  {format_type}: {filepath}")
        
        return saved_files
