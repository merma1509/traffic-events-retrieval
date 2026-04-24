import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class TrafficTextPreprocessor:
    """Text preprocessing with better token extraction"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.traffic_stop_words = self._get__stop_words()
        self.traffic_keywords = self._get_traffic_keywords()
        self.token_stats = {}
        
    def _get__stop_words(self) -> Set[str]:
        """Get refined stop words - keep more meaningful terms"""
        # Standard English stop words
        standard_stop = set(stopwords.words('english'))
        
        # Remove some terms that are meaningful for traffic IR
        meaningful_terms = {'weather', 'traffic', 'congestion', 'delays', 'conditions'}
        standard_stop = standard_stop - meaningful_terms
        
        # Traffic-specific stop words (only truly non-searchable ones)
        traffic_stop = {
            'event', 'road', 'from', 'to', 'with', 'degrees', 'km', 'meters',
            'vehicles', 'capacity', 'temperature', 'precipitation', 'mm', 'wind', 'speed', 'direction',
            'visibility', 'pressure', 'hpa', 'humidity', 'percent', 'lanes', 'limit', 'along',
            'expected', 'impact', 'severe', 'moderate', 'light', 'heavy',
            'delays', 'hazardous', 'driving', 'conditions', 'peak', 'during'
        }
        
        # Combine and convert to lowercase
        all_stop_words = standard_stop.union(traffic_stop)
        return {word.lower() for word in all_stop_words}
    
    def _get_traffic_keywords(self) -> Dict[str, List[str]]:
        """Get domain-specific keywords and their variants"""
        return {
            'congestion': ['congestion', 'congested', 'traffic jam', 'traffic backup', 'gridlock'],
            'weather': ['rain', 'heavy rain', 'storm', 'clear', 'sunny', 'cloudy', 'fog', 'mist'],
            'road_types': ['motorway', 'highway', 'primary', 'secondary', 'tertiary', 'residential', 'service', 'trunk'],
            'time_periods': ['morning', 'evening', 'night', 'daytime', 'rush', 'peak', 'off-peak'],
            'impacts': ['delays', 'slow', 'fast', 'blocked', 'closed', 'detour', 'accident'],
            'visibility': ['poor visibility', 'good visibility', 'limited visibility', 'clear visibility'],
            'wind': ['strong wind', 'light wind', 'calm', 'breezy', 'gusty']
        }
    
    def _clean_text(self, text: str) -> str:
        """ text cleaning that preserves meaningful terms"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Preserve meaningful compound terms before splitting
        text = re.sub(r'heavy rain', 'heavy_rain', text)
        text = re.sub(r'poor visibility', 'poor_visibility', text)
        text = re.sub(r'morning rush', 'morning_rush', text)
        text = re.sub(r'evening rush', 'evening_rush', text)
        text = re.sub(r'rush hour', 'rush_hour', text)
        text = re.sub(r'roundabout', 'roundabout', text)  # Keep as single term
        
        # Handle traffic-specific patterns
        text = re.sub(r'node_(\d+)', r'node_\1', text)                # Keep node_ prefix
        text = re.sub(r'(\d+\.?\d*)\s*km', r'\1_km', text)            # Distance units
        text = re.sub(r'(\d+)\s*meters', r'\1_meters', text)          # Length units
        text = re.sub(r'(\d+)\s*vehicles', r'\1_vehicles', text)      # Vehicle counts
        text = re.sub(r'(\d+)\.?\d*\s*degrees', r'\1_degrees', text)  # Temperature
        text = re.sub(r'(\d+)\.?\d*\s*mm', r'\1_mm', text)            # Precipitation
        text = re.sub(r'(\d+)\s*km/h', r'\1_kmh', text)               # Speed
        text = re.sub(r'(\d+)\s*hpa', r'\1_hpa', text)                # Pressure
        
        # Handle time patterns
        text = re.sub(r'(\d{4}-\d{2}-\d{2})', r'\1', text)  # Dates
        text = re.sub(r'(\d{2}:\d{2}:\d{2})', r'\1', text)  # Times
        
        # Remove punctuation except underscores and hyphens in compounds
        text = re.sub(r'[^\w\s\-_]', ' ', text)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_meaningful_tokens(self, text: str) -> List[str]:
        """Extract meaningful tokens using  patterns"""
        if not text:
            return []
        
        # Split into tokens
        tokens = text.split()
        meaningful_tokens = []
        
        for token in tokens:
            # Skip stop words
            if token.lower() in self.traffic_stop_words:
                continue
            
            # Skip pure numbers but keep meaningful number-based tokens
            if token.isdigit():
                continue
            
            # Skip very short tokens (except meaningful ones)
            if len(token) == 1 and token not in {'x', 'y', 'a', 'b'}:
                continue
            
            # Skip very long tokens (likely errors)
            if len(token) > 25:
                continue
            
            # Keep tokens with reasonable length
            if 2 <= len(token) <= 25:
                meaningful_tokens.append(token)
        
        return meaningful_tokens
    
    def extract_congestion_indicators(self, text: str) -> List[str]:
        """Extract congestion-related terms"""
        congestion_terms = []
        
        # Direct congestion mentions (case-insensitive)
        text_lower = text.lower()
        if 'heavy congestion' in text_lower:
            congestion_terms.append('heavy_congestion')
        elif 'moderate congestion' in text_lower:
            congestion_terms.append('moderate_congestion')
        elif 'light traffic' in text_lower:
            congestion_terms.append('light_traffic')
        elif 'free flow' in text_lower:
            congestion_terms.append('free_flow')
        
        # Additional patterns that might appear
        if 'heavy traffic' in text_lower:
            congestion_terms.append('heavy_traffic')
        elif 'moderate traffic' in text_lower:
            congestion_terms.append('moderate_traffic')
        elif 'traffic jam' in text_lower:
            congestion_terms.append('traffic_jam')
        elif 'congested' in text_lower:
            congestion_terms.append('congested')
        elif 'gridlock' in text_lower:
            congestion_terms.append('gridlock')
        
        # Impact indicators
        if 'severe delays' in text_lower:
            congestion_terms.append('severe_delays')
        if 'peak traffic' in text_lower:
            congestion_terms.append('peak_traffic')
        if 'roundabout congestion' in text_lower:
            congestion_terms.append('roundabout_congestion')
        
        return congestion_terms
    
    def extract_weather_indicators(self, text: str) -> List[str]:
        """Extract weather-related terms"""
        weather_terms = []
        
        # Weather conditions (case-insensitive)
        text_lower = text.lower()
        if 'heavy rain' in text_lower:
            weather_terms.append('heavy_rain')
        elif 'rain' in text_lower:
            weather_terms.append('rain')
        elif 'clear' in text_lower:
            weather_terms.append('clear')
        elif 'cold' in text_lower:
            weather_terms.append('cold')
        elif 'hot' in text_lower:
            weather_terms.append('hot')
        elif 'poor visibility' in text_lower:
            weather_terms.append('poor_visibility')
        elif 'storm' in text_lower:
            weather_terms.append('storm')
        elif 'fog' in text_lower or 'mist' in text_lower:
            weather_terms.append('fog')
        elif 'snow' in text_lower:
            weather_terms.append('snow')
        
        # Weather impacts
        if 'hazardous driving' in text_lower:
            weather_terms.append('hazardous_driving')
        if 'rain during rush' in text_lower:
            weather_terms.append('rain_rush_hour')
        
        return weather_terms
    
    def extract_spatial_indicators(self, text: str) -> List[str]:
        """Extract spatial and location-related terms"""
        spatial_terms = []
        
        # Node references
        node_pattern = r'node_(\d+)'
        nodes = re.findall(node_pattern, text)
        for node in nodes[:5]:  # Limit to prevent too many
            spatial_terms.append(f'node_{node}')
        
        # Road types
        road_types = ['motorway', 'primary', 'secondary', 'tertiary', 'residential', 'service', 'trunk']
        for road_type in road_types:
            if road_type in text:
                spatial_terms.append(road_type)
        
        # Route references
        route_pattern = r'route\s+(\w+)'
        routes = re.findall(route_pattern, text)
        spatial_terms.extend(routes)
        
        return spatial_terms
    
    def extract_temporal_indicators(self, text: str) -> List[str]:
        """Extract time-related terms"""
        temporal_terms = []
        
        # Time periods
        if 'morning rush' in text:
            temporal_terms.append('morning_rush')
        elif 'evening rush' in text:
            temporal_terms.append('evening_rush')
        elif 'rush hour' in text:
            temporal_terms.append('rush_hour')
        elif 'night' in text:
            temporal_terms.append('night')
        elif 'daytime' in text:
            temporal_terms.append('daytime')
        
        # Day types
        if 'weekend' in text:
            temporal_terms.append('weekend')
        elif 'weekday' in text:
            temporal_terms.append('weekday')
        elif 'friday' in text:
            temporal_terms.append('friday')
        
        return temporal_terms
    
    def extract_numerical_features(self, text: str) -> List[str]:
        """Extract meaningful numerical features"""
        numerical_terms = []
        
        # Vehicle counts (meaningful ranges)
        vehicle_pattern = r'(\d+)\s*vehicles'
        vehicles = re.findall(vehicle_pattern, text)
        for vehicle_count in vehicles:
            count = int(vehicle_count)
            if count > 500:
                numerical_terms.append('high_vehicle_count')
            elif count > 200:
                numerical_terms.append('moderate_vehicle_count')
            elif count > 50:
                numerical_terms.append('low_vehicle_count')
        
        # Speed limits
        speed_pattern = r'speed limit\s+(\d+)'
        speeds = re.findall(speed_pattern, text)
        for speed in speeds:
            speed_int = int(speed)
            if speed_int >= 80:
                numerical_terms.append('high_speed_limit')
            elif speed_int >= 50:
                numerical_terms.append('medium_speed_limit')
            else:
                numerical_terms.append('low_speed_limit')
        
        # Temperature ranges
        temp_pattern = r'temperature:\s*(\d+\.?\d*)'
        temps = re.findall(temp_pattern, text)
        for temp in temps:
            temp_float = float(temp)
            if temp_float > 30:
                numerical_terms.append('high_temperature')
            elif temp_float < 5:
                numerical_terms.append('low_temperature')
            else:
                numerical_terms.append('moderate_temperature')
        
        return numerical_terms
    
    def _tokenize(self, text: str) -> Dict[str, List[str]]:
        """Tokenization with multiple token categories"""
        cleaned_text = self._clean_text(text)
        
        # Extract different types of tokens
        base_tokens = self.extract_meaningful_tokens(cleaned_text)
        congestion_tokens = self.extract_congestion_indicators(text)
        weather_tokens = self.extract_weather_indicators(text)
        spatial_tokens = self.extract_spatial_indicators(text)
        temporal_tokens = self.extract_temporal_indicators(text)
        numerical_tokens = self.extract_numerical_features(text)
        
        # Combine all tokens
        all_tokens = list(set(base_tokens + congestion_tokens + weather_tokens + 
                             spatial_tokens + temporal_tokens + numerical_tokens))
        
        return {
            'base_tokens': base_tokens,
            'congestion_tokens': congestion_tokens,
            'weather_tokens': weather_tokens,
            'spatial_tokens': spatial_tokens,
            'temporal_tokens': temporal_tokens,
            'numerical_tokens': numerical_tokens,
            'all_tokens': all_tokens
        }
    
    def preprocess_document_(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Document preprocessing"""
        text = document.get('text', '')
        
        #  tokenization
        token_result = self._tokenize(text)
        
        # Create searchable text from all tokens
        searchable_text = ' '.join(token_result['all_tokens'])
        
        # Update document with  tokens
        processed_doc = document.copy()
        processed_doc.update({
            'cleaned_text': self._clean_text(text),
            'tokens': token_result['base_tokens'],
            'congestion_tokens': token_result['congestion_tokens'],
            'weather_tokens': token_result['weather_tokens'],
            'spatial_tokens': token_result['spatial_tokens'],
            'temporal_tokens': token_result['temporal_tokens'],
            'numerical_tokens': token_result['numerical_tokens'],
            'all_tokens': token_result['all_tokens'],
            'token_count': len(token_result['all_tokens']),
            'unique_tokens': len(set(token_result['all_tokens'])),
            'searchable_text': searchable_text
        })
        
        return processed_doc
    
    def preprocess_corpus_(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ corpus preprocessing"""
        print(f" preprocessing of {len(documents)} traffic event documents...")
        
        processed_documents = []
        token_counts = []
        category_counts = {
            'congestion': 0,
            'weather': 0,
            'spatial': 0,
            'temporal': 0,
            'numerical': 0
        }
        
        for i, doc in enumerate(documents):
            processed_doc = self.preprocess_document_(doc)
            processed_documents.append(processed_doc)
            token_counts.append(processed_doc['token_count'])
            
            # Track category usage
            if processed_doc['congestion_tokens']:
                category_counts['congestion'] += 1
            if processed_doc['weather_tokens']:
                category_counts['weather'] += 1
            if processed_doc['spatial_tokens']:
                category_counts['spatial'] += 1
            if processed_doc['temporal_tokens']:
                category_counts['temporal'] += 1
            if processed_doc['numerical_tokens']:
                category_counts['numerical'] += 1
            
            # Progress reporting
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1:,} documents...")
        
        #  statistics
        self.token_stats = {
            'total_documents': len(processed_documents),
            'total_tokens': sum(token_counts),
            'avg_tokens_per_doc': np.mean(token_counts),
            'median_tokens_per_doc': np.median(token_counts),
            'min_tokens_per_doc': min(token_counts),
            'max_tokens_per_doc': max(token_counts),
            'unique_tokens_in_corpus': len(set(token for doc in processed_documents for token in doc['all_tokens'])),
            'category_coverage': category_counts
        }
        
        print(f"\n Preprocessing Statistics:")
        print(f"Total documents: {self.token_stats['total_documents']:,}")
        print(f"Total tokens: {self.token_stats['total_tokens']:,}")
        print(f"Average tokens per document: {self.token_stats['avg_tokens_per_doc']:.1f}")
        print(f"Unique tokens in corpus: {self.token_stats['unique_tokens_in_corpus']:,}")
        print(f"\nToken Category Coverage:")
        for category, count in category_counts.items():
            percentage = count / len(processed_documents) * 100
            print(f"  {category}: {count:,} ({percentage:.1f}%)")
        
        return processed_documents
    
    def analyze__vocabulary(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze  vocabulary"""
        print("\nAnalyzing  vocabulary...")
        
        # Collect tokens by category
        all_base_tokens = []
        all_congestion = []
        all_weather = []
        all_spatial = []
        all_temporal = []
        all_numerical = []
        
        for doc in documents:
            all_base_tokens.extend(doc.get('base_tokens', []))
            all_congestion.extend(doc.get('congestion_tokens', []))
            all_weather.extend(doc.get('weather_tokens', []))
            all_spatial.extend(doc.get('spatial_tokens', []))
            all_temporal.extend(doc.get('temporal_tokens', []))
            all_numerical.extend(doc.get('numerical_tokens', []))
        
        # Analyze each category
        vocab_analysis = {
            'base_tokens': Counter(all_base_tokens).most_common(15),
            'congestion_tokens': Counter(all_congestion).most_common(10),
            'weather_tokens': Counter(all_weather).most_common(10),
            'spatial_tokens': Counter(all_spatial).most_common(10),
            'temporal_tokens': Counter(all_temporal).most_common(10),
            'numerical_tokens': Counter(all_numerical).most_common(10)
        }
        
        print(f" Token Analysis:")
        for category, tokens in vocab_analysis.items():
            if tokens:
                print(f"\n{category.replace('_', ' ').title()}:")
                for token, count in tokens[:5]:
                    print(f"  {token}: {count}")
        
        return vocab_analysis