"""Batch processing components for traffic events retrieval system"""

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Callable
import time
from functools import partial
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from data import TrafficEventDocumentGenerator, TrafficTextPreprocessor


class BatchProcessor:
    """Batch processing with parallel execution for large datasets"""
    
    def __init__(self, batch_size: int = 10000, n_workers: int = None):
        self.batch_size = batch_size
        self.n_workers = n_workers or min(cpu_count(), 8)  # Limit to 8 workers
        self.document_generator = TrafficEventDocumentGenerator()
        self.text_preprocessor = TrafficTextPreprocessor()
        
    def process_batch(self, batch_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process a single batch of traffic data"""
        try:
            # Generate documents for this batch
            corpus_batch = self.document_generator.generate_corpus(batch_data)
            
            # Preprocess documents for this batch
            processed_batch = self.text_preprocessor.preprocess_corpus_(corpus_batch)
            
            return processed_batch
        except Exception as e:
            print(f"Error processing batch: {e}")
            return []
    
    def process_in_batches_parallel(self, data: pd.DataFrame, progress_callback: Callable = None) -> List[Dict[str, Any]]:
        """Process entire dataset in parallel batches"""
        total_rows = len(data)
        n_batches = (total_rows + self.batch_size - 1) // self.batch_size
        
        print(f"Processing {total_rows:,} records in {n_batches} batches")
        print(f"Batch size: {self.batch_size:,}")
        print(f"Workers: {self.n_workers}")
        
        # Split data into batches
        batches = [data.iloc[i:i + self.batch_size] for i in range(0, total_rows, self.batch_size)]
        
        # Process batches in parallel
        start_time = time.time()
        all_results = []
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all batches
            future_to_batch = {executor.submit(self.process_batch, batch): i for i, batch in enumerate(batches)}
            
            # Collect results as they complete
            completed = 0
            for future in future_to_batch:
                try:
                    batch_result = future.result()
                    all_results.extend(batch_result)
                    completed += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(completed, n_batches, len(all_results))
                    elif completed % max(1, n_batches // 10) == 0:  # Show progress every 10%
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (n_batches - completed) / rate if rate > 0 else 0
                        print(f"Progress: {completed}/{n_batches} batches ({completed/n_batches*100:.1f}%) - "
                              f"Documents: {len(all_results):,} - Rate: {rate:.1f} batches/sec - ETA: {eta:.0f}s")
                
                except Exception as e:
                    batch_idx = future_to_batch[future]
                    print(f"Batch {batch_idx} failed: {e}")
        
        total_time = time.time() - start_time
        print(f"\nCompleted processing in {total_time:.1f} seconds")
        print(f"Total documents processed: {len(all_results):,}")
        print(f"Average processing rate: {len(all_results)/total_time:.1f} documents/sec")
        
        return all_results
    
    def process_sample_fast(self, data: pd.DataFrame, sample_size: int = 10000) -> List[Dict[str, Any]]:
        """Process a smaller sample for quick testing"""
        print(f"Processing sample of {sample_size:,} records...")
        
        # Take random sample
        if len(data) > sample_size:
            sample_data = data.sample(n=sample_size, random_state=42)
        else:
            sample_data = data
        
        # Process sample in parallel
        return self.process_in_batches_parallel(sample_data)
    
    def estimate_processing_time(self, data: pd.DataFrame) -> Dict[str, float]:
        """Estimate processing time based on a small sample"""
        print("Estimating processing time...")
        
        # Take a small sample for timing
        sample_size = min(1000, len(data))
        sample_data = data.sample(n=sample_size, random_state=42)
        
        start_time = time.time()
        sample_result = self.process_batch(sample_data)
        sample_time = time.time() - start_time
        
        # Estimate total time
        records_per_second = sample_size / sample_time
        total_records = len(data)
        estimated_total_time = total_records / records_per_second
        
        return {
            "sample_size": sample_size,
            "sample_time": sample_time,
            "records_per_second": records_per_second,
            "total_records": total_records,
            "estimated_total_time": estimated_total_time,
            "estimated_time_minutes": estimated_total_time / 60,
            "estimated_time_hours": estimated_total_time / 3600
        }


def progress_callback(completed: int, total: int, documents: int):
    """Simple progress callback"""
    percentage = completed / total * 100
    print(f"Progress: {completed}/{total} batches ({percentage:.1f}%) - Documents: {documents:,}")
