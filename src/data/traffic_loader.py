import os
import pandas as pd
from pathlib import Path
from typing import Dict

class TrafficWeatherDataLoader:
    """Load and analyze traffic-weather temporal data"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.traffic_data = None
        self.data_stats = {}
        
    @staticmethod    
    def get_data_path(filename):
        """Get data path with multiple fallback options"""
        possible_paths = [
            f"./data/{filename}",
            f"../data/{filename}",
            f"../../data/{filename}",
            f"{os.getcwd()}/data/{filename}",
            filename,
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found data at: {path}")
                return path
        
        print(f"Data file not found: {filename}")
        print("Please ensure the data file is available")
        return filename
        
    def load_traffic_weather_data(self, filename: str = "traffic_weather_temporal.csv") -> pd.DataFrame:
        """Load the main traffic-weather dataset"""
        print("Loading Traffic-Weather Temporal Data...")
        
        # Use the path resolution method
        file_path = self.get_data_path(filename)
        file_path = Path(file_path)  # Convert to Path object
        
        try:
            # Load with optimized parameters for large dataset
            self.traffic_data = pd.read_csv(
                file_path,
                parse_dates=['timestamp', 'datetime'],
                dtype={
                    'source_node': 'int32',
                    'target_node': 'int32',
                    'day_of_week': 'int8',
                    'hour_of_day': 'int8',
                    'weather_code': 'int8',
                    'is_rain': 'int8',
                    'is_heavy_rain': 'int8',
                    'is_weekend': 'bool',
                    'is_rush_hour': 'bool'
                }
            )
            
            print(f"Loaded {len(self.traffic_data):,} traffic records")
            print(f"Time range: {self.traffic_data['timestamp'].min()} to {self.traffic_data['timestamp'].max()}")
            print(f"Network: {self.traffic_data['source_node'].nunique()} source nodes, {self.traffic_data['target_node'].nunique()} target nodes")
            
            return self.traffic_data
            
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            print(f"Attempted path: {file_path}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def analyze_data_quality(self) -> Dict:
        """Analyze data quality and generate statistics"""
        if self.traffic_data is None:
            raise ValueError("Data not loaded. Call load_traffic_weather_data() first.")
        
        print("\nData Quality Analysis...")
        
        stats = {
            'total_rows': len(self.traffic_data),
            'total_columns': len(self.traffic_data.columns),
            'memory_usage_mb': self.traffic_data.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': self.traffic_data.isnull().sum().sum(),
            'duplicate_rows': self.traffic_data.duplicated().sum(),
            'date_range': {
                'start': self.traffic_data['timestamp'].min(),
                'end': self.traffic_data['timestamp'].max(),
                'hours': (self.traffic_data['timestamp'].max() - self.traffic_data['timestamp'].min()).total_seconds() / 3600
            },
            'network_stats': {
                'unique_sources': self.traffic_data['source_node'].nunique(),
                'unique_targets': self.traffic_data['target_node'].nunique(),
                'unique_pairs': self.traffic_data[['source_node', 'target_node']].drop_duplicates().shape[0]
            },
            'weather_stats': {
                'rainy_records': self.traffic_data[self.traffic_data['is_rain'] == 1].shape[0],
                'heavy_rain_records': self.traffic_data[self.traffic_data['is_heavy_rain'] == 1].shape[0],
                'avg_temperature': self.traffic_data['temperature'].mean(),
                'avg_precipitation': self.traffic_data['precipitation'].mean()
            },
            'traffic_stats': {
                'rush_hour_records': self.traffic_data[self.traffic_data['is_rush_hour'] == True].shape[0],
                'weekend_records': self.traffic_data[self.traffic_data['is_weekend'] == True].shape[0],
                'avg_vehicle_count': self.traffic_data['vehicle_counts'].mean(),
                'max_vehicle_count': self.traffic_data['vehicle_counts'].max()
            }
        }
        
        self.data_stats = stats
        
        # Print summary
        print(f"Dataset Size: {stats['total_rows']:,} rows × {stats['total_columns']} columns")
        print(f"Memory Usage: {stats['memory_usage_mb']:.1f} MB")
        print(f"Missing Values: {stats['missing_values']}")
        print(f"Duplicate Rows: {stats['duplicate_rows']}")
        print(f"Time Coverage: {stats['date_range']['hours']:.1f} hours")
        print(f"Rainy Records: {stats['weather_stats']['rainy_records']:,} ({stats['weather_stats']['rainy_records']/stats['total_rows']*100:.1f}%)")
        print(f"Rush Hour Records: {stats['traffic_stats']['rush_hour_records']:,}")
        
        return stats
    
    def get_sample_data(self, n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
        """Get a sample of the data for development"""
        if self.traffic_data is None:
            raise ValueError("Data not loaded. Call load_traffic_weather_data() first.")
        
        if len(self.traffic_data) <= n_samples:
            return self.traffic_data.copy()
        
        return self.traffic_data.sample(n=n_samples, random_state=random_state)