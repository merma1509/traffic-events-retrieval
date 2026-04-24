"""Query processing components for traffic events retrieval system"""

import re
from typing import List, Dict, Any, Optional, Set
from collections import Counter
import json


class QueryProcessor:
    """Advanced query processing for traffic event retrieval"""
    
    def __init__(self):
        self.stop_words = self._get_stop_words()
        self.traffic_keywords = self._get_traffic_keywords()
        
    def _get_stop_words(self) -> Set[str]:
        """Get stop words for query processing"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
    
    def _get_traffic_keywords(self) -> Dict[str, List[str]]:
        """Get traffic-specific keyword mappings"""
        return {
            'congestion': ['congestion', 'traffic', 'jam', 'delays', 'heavy', 'moderate', 'light', 'severe', 'gridlock'],
            'weather': ['rain', 'weather', 'storm', 'clear', 'sunny', 'fog', 'mist', 'snow', 'cold', 'hot', 'precipitation'],
            'spatial': ['road', 'highway', 'street', 'motorway', 'node', 'route', 'junction', 'intersection', 'bridge'],
            'temporal': ['morning', 'evening', 'rush', 'hour', 'day', 'night', 'time', 'weekend', 'weekday'],
            'vehicle': ['vehicle', 'car', 'truck', 'bus', 'motorcycle', 'lanes', 'capacity']
        }
    
    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Preprocess user query for better matching
        
        Args:
            query: Raw user query string
            
        Returns:
            Dictionary with processed query components
        """
        # Basic cleaning
        query = query.lower().strip()
        query = re.sub(r'[^\w\s]', ' ', query)  # Keep only alphanumeric and spaces
        query = re.sub(r'\s+', ' ', query)  # Normalize whitespace
        
        # Tokenize
        tokens = query.split()
        
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
        
        # Analyze query intent
        query_analysis = self._analyze_query_intent(tokens)
        
        # Extract query features
        query_features = self._extract_query_features(tokens)
        
        return {
            'original_query': query,
            'tokens': tokens,
            'filtered_tokens': tokens,  # After stop word removal
            'intent_analysis': query_analysis,
            'features': query_features,
            'query_type': query_analysis.get('primary_intent', 'general'),
            'expanded_terms': self._expand_query_terms(tokens)
        }
    
    def _analyze_query_intent(self, tokens: List[str]) -> Dict[str, Any]:
        """Analyze the intent behind the query"""
        intent_scores = {
            'congestion': 0,
            'weather': 0,
            'spatial': 0,
            'temporal': 0,
            'vehicle': 0
        }
        
        for token in tokens:
            for category, keywords in self.traffic_keywords.items():
                if token in keywords:
                    intent_scores[category] += 1
        
        # Determine primary intent
        if max(intent_scores.values()) > 0:
            primary_intent = max(intent_scores, key=intent_scores.get)
        else:
            primary_intent = 'general'
        
        return {
            'intent_scores': intent_scores,
            'primary_intent': primary_intent,
            'confidence': max(intent_scores.values()) / len(tokens) if tokens else 0
        }
    
    def _extract_query_features(self, tokens: List[str]) -> Dict[str, Any]:
        """Extract features from query tokens"""
        features = {
            'has_numbers': any(token.isdigit() for token in tokens),
            'has_locations': any(token in self.traffic_keywords['spatial'] for token in tokens),
            'has_time_refs': any(token in self.traffic_keywords['temporal'] for token in tokens),
            'has_weather': any(token in self.traffic_keywords['weather'] for token in tokens),
            'has_congestion': any(token in self.traffic_keywords['congestion'] for token in tokens),
            'query_length': len(tokens),
            'unique_tokens': len(set(tokens)),
            'token_frequency': Counter(tokens)
        }
        
        return features
    
    def _expand_query_terms(self, tokens: List[str]) -> List[str]:
        """Expand query terms with synonyms and related terms"""
        expanded = set(tokens)
        
        # Add traffic-specific expansions
        expansions = {
            'traffic': ['congestion', 'jam', 'delays'],
            'congestion': ['traffic', 'jam', 'heavy', 'moderate', 'light'],
            'rain': ['weather', 'precipitation', 'storm'],
            'road': ['highway', 'street', 'route', 'junction'],
            'morning': ['rush', 'hour', 'time'],
            'evening': ['rush', 'hour', 'time'],
            'heavy': ['severe', 'high', 'major'],
            'light': ['low', 'minor', 'free'],
            'clear': ['sunny', 'good', 'fair']
        }
        
        for token in tokens:
            if token in expansions:
                expanded.update(expansions[token])
        
        return list(expanded)
    
    def format_query_for_search(self, query_data: Dict[str, Any]) -> List[str]:
        """Format processed query for different search strategies"""
        tokens = query_data['tokens']
        expanded_terms = query_data['expanded_terms']
        intent = query_data['intent_analysis']['primary_intent']
        
        # Different query formats based on intent
        if intent == 'congestion':
            # Focus on congestion-related terms
            search_terms = [term for term in expanded_terms if term in self.traffic_keywords['congestion']]
        elif intent == 'weather':
            # Focus on weather-related terms
            search_terms = [term for term in expanded_terms if term in self.traffic_keywords['weather']]
        elif intent == 'spatial':
            # Focus on location-related terms
            search_terms = [term for term in expanded_terms if term in self.traffic_keywords['spatial']]
        else:
            # General search - use all expanded terms
            search_terms = expanded_terms
        
        return search_terms
    
    def generate_query_variations(self, query: str) -> List[Dict[str, Any]]:
        """Generate multiple query variations for better recall"""
        variations = []
        
        # Original query
        original = self.preprocess_query(query)
        variations.append({
            'type': 'original',
            'query_data': original,
            'description': 'Original query'
        })
        
        # Expanded query
        expanded = self.preprocess_query(query)
        variations.append({
            'type': 'expanded',
            'query_data': expanded,
            'description': 'Expanded with synonyms'
        })
        
        # Phrase-based variations
        tokens = query.lower().split()
        if len(tokens) > 1:
            # Try different phrase combinations
            for i in range(len(tokens)):
                for j in range(i + 1, len(tokens) + 1):
                    phrase = ' '.join(tokens[i:j])
                    if len(phrase.split()) > 1:  # Only meaningful phrases
                        phrase_data = self.preprocess_query(phrase)
                        variations.append({
                            'type': 'phrase',
                            'query_data': phrase_data,
                            'description': f'Phrase: "{phrase}"'
                        })
        
        # Boolean variations
        if 'and' in query.lower() or 'or' in query.lower():
            # Split boolean queries
            boolean_parts = re.split(r'\s+(and|or)\s+', query.lower())
            for i, part in enumerate(boolean_parts):
                part_data = self.preprocess_query(part.strip())
                variations.append({
                    'type': 'boolean_part',
                    'query_data': part_data,
                    'description': f'Boolean part {i+1}: "{part.strip()}"'
                })
        
        return variations
    
    def get_query_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """Get query suggestions based on common traffic queries"""
        common_queries = [
            'heavy congestion',
            'traffic jam',
            'rain weather',
            'road closure',
            'rush hour traffic',
            'clear weather conditions',
            'accident report',
            'construction delays',
            'public transportation',
            'highway conditions'
        ]
        
        query_lower = query.lower()
        suggestions = []
        
        for common_query in common_queries:
            if query_lower in common_query or common_query in query_lower:
                suggestions.append(common_query)
        
        # Add partial matches
        for common_query in common_queries:
            common_tokens = set(common_query.split())
            query_tokens = set(query_lower.split())
            
            # Check for partial overlap
            overlap = len(common_tokens & query_tokens)
            if overlap > 0 and overlap < len(query_tokens):
                suggestions.append(common_query)
        
        # Remove duplicates and limit
        suggestions = list(dict.fromkeys(suggestions))  # Preserve order, remove duplicates
        return suggestions[:limit]
    
    def explain_query_processing(self, query_data: Dict[str, Any]) -> str:
        """Generate explanation of query processing"""
        explanation = []
        
        explanation.append(f"Original Query: '{query_data['original_query']}'")
        explanation.append(f"Tokens: {query_data['tokens']}")
        explanation.append(f"Filtered Tokens: {query_data['filtered_tokens']}")
        
        intent = query_data['intent_analysis']
        explanation.append(f"Primary Intent: {intent['primary_intent']} (confidence: {intent['confidence']:.2f})")
        
        if intent['primary_intent'] != 'general':
            explanation.append(f"Intent Scores: {intent['intent_scores']}")
        
        features = query_data['features']
        if features['has_numbers']:
            explanation.append("Contains numeric values")
        if features['has_locations']:
            explanation.append("Contains location references")
        if features['has_time_refs']:
            explanation.append("Contains time references")
        if features['has_weather']:
            explanation.append("Contains weather terms")
        if features['has_congestion']:
            explanation.append("Contains congestion terms")
        
        if query_data['expanded_terms'] != query_data['tokens']:
            explanation.append(f"Expanded Terms: {query_data['expanded_terms']}")
        
        return '\n'.join(explanation)
