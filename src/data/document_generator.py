import math
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .network_loader import KigaliNetworkLoader

class TrafficEventDocumentGenerator:
    """Transform traffic data rows into searchable IR documents"""
    
    def __init__(self, network_loader: Optional['KigaliNetworkLoader'] = None):
        self.network_loader = network_loader
        # Only build network lookup if provided and needed
        if network_loader and network_loader.network_graph:
            self.edge_attributes = self._build_edge_attribute_lookup()
        else:
            self.edge_attributes = {}
    
    def _build_edge_attribute_lookup(self) -> Dict:
        """Build lookup table for edge attributes"""
        if not self.network_loader or not self.network_loader.network_graph:
            return {}
        
        edge_lookup = {}
        G = self.network_loader.network_graph
        
        for u, v, data in G.edges(data=True):
            edge_key = f"{u}_{v}"
            edge_lookup[edge_key] = {
                'highway': data.get('highway', 'unknown'),
                'name': data.get('name', ''),
                'oneway': data.get('oneway', False),
                'length': data.get('length', 0),
                'lanes': data.get('lanes', ''),
                'maxspeed': data.get('maxspeed', ''),
                'junction': data.get('junction', ''),
                'service': data.get('service', ''),
                'ref': data.get('ref', ''),
                'geometry': data.get('geometry', None)
            }
        
        print(f"Built network edge attribute lookup for {len(edge_lookup)} edges (optional enhancement)")
        return edge_lookup
    
    def _get__attributes(self, source_node: int, target_node: int) -> Dict:
        """Get network attributes if available, otherwise return empty"""
        if not self.edge_attributes:
            return {}
        
        edge_key = f"{source_node}_{target_node}"
        return self.edge_attributes.get(edge_key, {})
    
    def _determine_congestion_level(self, vehicle_count: float, road_capacity: int) -> str:
        """Determine congestion level based on vehicle count and capacity"""
        if road_capacity > 0:
            utilization = vehicle_count / road_capacity
        else:
            utilization = 0
        
        if utilization > 0.8 or vehicle_count > 500:
            return "Heavy Congestion"
        elif utilization > 0.5 or vehicle_count > 200:
            return "Moderate Congestion"
        elif utilization > 0.2 or vehicle_count > 50:
            return "Light Traffic"
        else:
            return "Free Flow"
    
    def _classify_weather_condition(self, row: pd.Series) -> str:
        """Classify weather condition using traffic data features"""
        is_rain = row.get('is_rain', 0)
        is_heavy_rain = row.get('is_heavy_rain', 0)
        precipitation = row.get('precipitation', 0)
        temperature = row.get('temperature', 20)
        visibility = row.get('visibility', 10)
        is_hot = row.get('is_hot', 0)
        is_cold = row.get('is_cold', 0)
        
        if is_heavy_rain == 1 or precipitation > 5.0:
            return "Heavy Rain"
        elif is_rain == 1 or precipitation > 0.1:
            return "Rain"
        elif is_cold == 1 or temperature < 5:
            return "Cold"
        elif is_hot == 1 or temperature > 30:
            return "Hot"
        elif visibility < 5:
            return "Poor Visibility"
        else:
            return "Clear"
    
    def _get_time_context(self, row: pd.Series) -> str:
        """Get time-based context using traffic data features"""
        hour_of_day = int(row['hour_of_day'])
        day_of_week = int(row['day_of_week'])
        is_rush_hour = bool(row['is_rush_hour'])
        is_weekend = bool(row['is_weekend'])
        
        time_contexts = []
        
        if 6 <= hour_of_day <= 9:
            time_contexts.append("Morning Rush")
        elif 17 <= hour_of_day <= 19:
            time_contexts.append("Evening Rush")
        elif 22 <= hour_of_day <= 24 or 0 <= hour_of_day <= 5:
            time_contexts.append("Night")
        elif 10 <= hour_of_day <= 16:
            time_contexts.append("Daytime")
        
        if is_weekend:
            time_contexts.append("Weekend")
        elif day_of_week >= 5:
            time_contexts.append("Friday")
        else:
            time_contexts.append("Weekday")
        
        if is_rush_hour:
            time_contexts.append("Rush Hour")
        
        return " ".join(time_contexts)
    
    def _calculate_distance_from_coordinates(self, row: pd.Series) -> float:
        """Calculate distance using traffic data coordinates"""
        source_x = float(row['source_x'])
        source_y = float(row['source_y'])
        target_x = float(row['target_x'])
        target_y = float(row['target_y'])
        
        # Calculate Euclidean distance and convert to approximate km
        distance = math.sqrt((target_x - source_x)**2 + (target_y - source_y)**2)
        return distance * 111  # Rough conversion to kilometers
    
    def _get_weather_impact_level(self, row: pd.Series) -> str:
        """Determine weather impact level using traffic data features"""
        is_rain = row.get('is_rain', 0)
        is_heavy_rain = row.get('is_heavy_rain', 0)
        precipitation = row.get('precipitation', 0)
        visibility = row.get('visibility', 10)
        wind_speed = row.get('wind_speed', 0)
        
        impact_factors = []
        
        if is_heavy_rain == 1:
            impact_factors.append("Severe Rain Impact")
        elif is_rain == 1 and precipitation > 2.0:
            impact_factors.append("Moderate Rain Impact")
        elif is_rain == 1:
            impact_factors.append("Light Rain Impact")
        
        if visibility < 1:
            impact_factors.append("Very Poor Visibility")
        elif visibility < 5:
            impact_factors.append("Poor Visibility")
        
        if wind_speed > 20:
            impact_factors.append("High Wind Conditions")
        
        if not impact_factors:
            return "Normal Weather Conditions"
        
        return ", ".join(impact_factors)
    
    def create_event_document(self, row: pd.Series, doc_id: str) -> Dict[str, Any]:
        """Transform a single traffic row into a searchable document"""
        
        # Extract key information
        source_node = int(row['source_node'])
        target_node = int(row['target_node'])
        timestamp = row['timestamp']
        vehicle_count = float(row['vehicle_counts'])
        highway_type = str(row['highway_type'])
        road_capacity = int(row['road_capacity'])
        road_length = float(row['road_length_meters'])
        speed_limit = row.get('speed_limit_kmh', 'unknown')
        lanes = row.get('lanes', 'unknown')
        
        # Weather features
        temperature = float(row['temperature'])
        precipitation = float(row['precipitation'])
        humidity = float(row['humidity'])
        pressure = float(row['pressure'])
        wind_speed = float(row['wind_speed'])
        wind_direction = float(row['wind_direction'])
        visibility = float(row['visibility'])
        cloud_cover = float(row['cloud_cover'])
        
        # Time features
        hour_of_day = int(row['hour_of_day'])
        day_of_week = int(row['day_of_week'])
        is_rush_hour = bool(row['is_rush_hour'])
        is_weekend = bool(row['is_weekend'])
        
        # Boolean weather flags
        is_rain = bool(row['is_rain'])
        is_heavy_rain = bool(row['is_heavy_rain'])
        is_hot = bool(row['is_hot'])
        is_cold = bool(row['is_cold'])
        
        # Coordinate features
        source_x = float(row['source_x'])
        source_y = float(row['source_y'])
        target_x = float(row['target_x'])
        target_y = float(row['target_y'])
        
        #  features
        rain_rush_hour = bool(row['rain_rush_hour'])
        rain_weekend = bool(row['rain_weekend'])
        temperature_lag_1h = float(row['temperature_lag_1h'])
        precipitation_lag_1h = float(row['precipitation_lag_1h'])
        
        # Get optional network attributes
        network_attrs = self._get__attributes(source_node, target_node)
        
        road_name = network_attrs.get('name', '')  # Only from network
        network_highway = network_attrs.get('highway', highway_type)  # with network if available
        maxspeed = network_attrs.get('maxspeed', speed_limit)  # with network if available
        junction = network_attrs.get('junction', '')  # Only from network
        service = network_attrs.get('service', '')  # Only from network
        ref = network_attrs.get('ref', '')  # Only from network
        
        # Determine event characteristics
        congestion_level = self._determine_congestion_level(vehicle_count, road_capacity)
        weather_condition = self._classify_weather_condition(row)
        time_context = self._get_time_context(row)
        distance_km = self._calculate_distance_from_coordinates(row)
        weather_impact = self._get_weather_impact_level(row)
        
        # Create comprehensive searchable event text
        event_text_parts = [
            f"{congestion_level} traffic event on {network_highway} road",
            f"from node {source_node} to node {target_node}"
        ]
        
        # Add road name if available from network
        if road_name:
            event_text_parts.append(f"along {road_name}")
        
        # Add road characteristics from traffic data
        road_features = []
        if lanes != 'unknown' and lanes:
            road_features.append(f"{lanes} lanes")
        if maxspeed != 'unknown' and str(maxspeed) != 'nan':
            road_features.append(f"speed limit {maxspeed}")
        if junction:
            road_features.append(f"near {junction}")
        if service:
            road_features.append(f"{service} road")
        if ref:
            road_features.append(f"route {ref}")
        
        if road_features:
            event_text_parts.append(f"({', '.join(road_features)})")
        
        # Add comprehensive weather information
        event_text_parts.append(
            f"Weather: {weather_condition} with {precipitation:.2f}mm precipitation"
        )
        event_text_parts.append(f"Temperature: {temperature:.1f} degrees (humidity: {humidity:.0f}%)")
        event_text_parts.append(f"Wind: {wind_speed:.1f} km/h from {wind_direction:.0f} degrees")
        event_text_parts.append(f"Visibility: {visibility:.1f} km, Pressure: {pressure:.0f} hPa")
        
        # Add traffic information
        event_text_parts.append(
            f"Traffic: {vehicle_count:.0f} vehicles on road with capacity {road_capacity}"
        )
        
        # Add spatial information
        if road_length > 0:
            event_text_parts.append(f"Road length: {road_length:.0f} meters")
        if distance_km > 0:
            event_text_parts.append(f"Distance: {distance_km:.2f} km")
        
        # Add temporal context
        event_text_parts.append(f"Time: {timestamp}. {time_context}")
        
        # Add impact indicators
        impact_indicators = []
        if congestion_level == "Heavy Congestion":
            impact_indicators.append("Severe delays expected")
        if weather_condition in ["Heavy Rain", "Poor Visibility"]:
            impact_indicators.append("Hazardous driving conditions")
        if is_rush_hour:
            impact_indicators.append("Peak traffic congestion")
        if rain_rush_hour:
            impact_indicators.append("Rain during rush hour")
        if rain_weekend:
            impact_indicators.append("Weekend rain conditions")
        if junction == "roundabout":
            impact_indicators.append("Roundabout congestion")
        if service in ["alley", "driveway"]:
            impact_indicators.append("Limited access road")
        
        # Combine all parts
        event_text = ". ".join(event_text_parts) + "."
        
        if impact_indicators:
            event_text += f" Impact: {' '.join(impact_indicators)}."
        
        # Create comprehensive document metadata
        document = {
            'doc_id': doc_id,
            'title': f"{congestion_level} - {weather_condition} - {network_highway}",
            'text': event_text,
            'timestamp': timestamp,
            'datetime': row['datetime'],
            
            # Node and spatial information
            'source_node': source_node,
            'target_node': target_node,
            'source_x': source_x,
            'source_y': source_y,
            'target_x': target_x,
            'target_y': target_y,
            'distance_km': distance_km,
            
            # Traffic characteristics
            'vehicle_count': vehicle_count,
            'congestion_level': congestion_level,
            'highway_type': network_highway,
            'road_name': road_name,
            'road_capacity': road_capacity,
            'road_length_meters': road_length,
            'speed_limit': speed_limit,
            'lanes': lanes,
            
            # Weather characteristics
            'weather_condition': weather_condition,
            'weather_impact': weather_impact,
            'temperature': temperature,
            'temperature_lag_1h': temperature_lag_1h,
            'precipitation': precipitation,
            'precipitation_lag_1h': precipitation_lag_1h,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'visibility': visibility,
            'cloud_cover': cloud_cover,
            
            # Weather flags
            'is_rain': is_rain,
            'is_heavy_rain': is_heavy_rain,
            'is_hot': is_hot,
            'is_cold': is_cold,
            
            # Temporal characteristics
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'time_context': time_context,
            'is_rush_hour': is_rush_hour,
            'is_weekend': is_weekend,
            'rain_rush_hour': rain_rush_hour,
            'rain_weekend': rain_weekend,
            
            # Other features
            'day_of_week_num': int(row['day_of_week_num']),
            'hour_sin': float(row['hour_sin']),
            'hour_cos': float(row['hour_cos']),
            'day_sin': float(row['day_sin']),
            'day_cos': float(row['day_cos']),
            'weather_code': int(row['weather_code']),
            'segment_multiplier': float(row['segment_multiplier']),
            
            # Network (if available)
            'network_': len(network_attrs) > 0,
            'junction': junction,
            'service': service,
            'ref': ref,
            'maxspeed': maxspeed,
            
            # Impact indicators
            'impact_indicators': impact_indicators,
            
            # Searchable fields for IR
            'location_tokens': f"node_{source_node} node_{target_node} {network_highway} {road_name} {ref}",
            'weather_tokens': f"{weather_condition} rain precipitation temperature {temperature:.0f} humidity {humidity:.0f} wind {wind_speed:.0f}",
            'traffic_tokens': f"{congestion_level} congestion vehicles {vehicle_count:.0f} capacity {road_capacity} rush_hour {is_rush_hour}",
            'time_tokens': f"hour_{hour_of_day} day_{day_of_week} {time_context.lower()} weekend {is_weekend}",
            'road_tokens': f"{network_highway} {lanes} {maxspeed} {junction} {service} {road_name} speed_limit {speed_limit}",
            'impact_tokens': f"{' '.join(impact_indicators).lower()}",
            'coordinate_tokens': f"source_{source_x:.6f}_{source_y:.6f} target_{target_x:.6f}_{target_y:.6f}"
        }
        
        return document
    
    def generate_corpus(self, traffic_data: pd.DataFrame, sample_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate complete IR document corpus from traffic data"""
        print("Generating Traffic Event Corpus from Traffic Data Features...")
        print(f"Available features: {list(traffic_data.columns)}")
        
        # Sample data if specified
        if sample_size and len(traffic_data) > sample_size:
            working_data = traffic_data.sample(n=sample_size, random_state=42)
            print(f"Using sample of {sample_size:,} records from {len(traffic_data):,} total")
        else:
            working_data = traffic_data
            print(f"Processing all {len(working_data):,} records")
        
        documents = []
        processed_count = 0
        network__count = 0
        
        for idx, row in working_data.iterrows():
            doc_id = f"traffic_event_{idx}"
            
            try:
                document = self.create_event_document(row, doc_id)
                
                # Track network enhancements
                if document.get('network_', False):
                    network__count += 1
                
                documents.append(document)
                processed_count += 1
                
                # Progress reporting
                if processed_count % 10000 == 0:
                    print(f"Processed {processed_count:,} documents...")
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        print(f"Generated {len(documents):,} traffic event documents")
        if self.edge_attributes:
            print(f"Network- documents: {network__count:,} ({network__count/len(documents)*100:.1f}%)")
        else:
            print("Using traffic data features only")
        
        return documents
