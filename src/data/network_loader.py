import os
import pickle
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

class KigaliNetworkLoader:
    """Load and analyze Kigali road network data"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.network_graph = None
        self.network_stats = {}

    @staticmethod    
    def get_data_path(filename: str) -> str:
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
    
    def load_network_graph(self, filename: str = 'kigali_congested_network.pkl') -> nx.MultiDiGraph:
        """Load the Kigali congested network graph"""
        print("Loading Kigali Road Network Graph...")
        
        file_path = self.get_data_path(filename)
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'rb') as f:
                self.network_graph = pickle.load(f)
            
            print(f"Loaded road network with {self.network_graph.number_of_nodes():,} nodes and {self.network_graph.number_of_edges():,} edges")
            print(f"Graph Type: {type(self.network_graph).__name__}")
            print(f"Connected Components: {nx.number_weakly_connected_components(self.network_graph)}")
            
            return self.network_graph
            
        except FileNotFoundError:
            print(f"Error: Network file not found at {file_path}")
            raise
        except Exception as e:
            print(f"Error loading network: {e}")
            raise
    
    def analyze_network_structure(self) -> Dict[str, Any]:
        """Analyze network topology and structure"""
        if self.network_graph is None:
            raise ValueError("Network not loaded. Call load_network_graph() first.")
        
        print("\nNetwork Structure Analysis...")
        
        G = self.network_graph
        
        stats = {
            'basic_info': {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'is_directed': G.is_directed(),
                'is_multigraph': G.is_multigraph(),
                'density': nx.density(G)
            },
            'connectivity': {
                'weakly_connected_components': nx.number_weakly_connected_components(G),
                'strongly_connected_components': nx.number_strongly_connected_components(G),
                'is_weakly_connected': nx.is_weakly_connected(G),
                'is_strongly_connected': nx.is_strongly_connected(G)
            },
            'degree_stats': {
                'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
                'max_degree': max(dict(G.degree()).values()),
                'min_degree': min(dict(G.degree()).values()),
                'avg_in_degree': sum(dict(G.in_degree()).values()) / G.number_of_nodes(),
                'avg_out_degree': sum(dict(G.out_degree()).values()) / G.number_of_nodes()
            },
            'node_attributes': self._analyze_node_attributes(),
            'edge_attributes': self._analyze_edge_attributes()  
        }
        
        self.network_stats = stats
        
        print(f"Network: {stats['basic_info']['nodes']:,} nodes, {stats['basic_info']['edges']:,} edges")
        print(f"Density: {stats['basic_info']['density']:.4f}")
        print(f"Avg Degree: {stats['degree_stats']['avg_degree']:.2f}")
        print(f"Connected Components: {stats['connectivity']['weakly_connected_components']}")
        
        return stats
    
    def _analyze_node_attributes(self) -> Dict[str, Any]:
        """Analyze node attributes"""
        G = self.network_graph
        
        if not G.nodes():
            return {}
        
        sample_node = list(G.nodes())[0]
        node_attrs = list(G.nodes[sample_node].keys())
        
        attr_analysis = {'available_attributes': node_attrs}
        
        for attr in node_attrs:
            values = [G.nodes[node].get(attr) for node in G.nodes() if attr in G.nodes[node]]
            
            if values and isinstance(values[0], (int, float)):
                attr_analysis[attr] = {
                    'type': 'numeric',
                    'count': len(values),
                    'mean': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                attr_analysis[attr] = {
                    'type': 'categorical',
                    'count': len(values),
                    'unique': len(set(values))
                }
        
        return attr_analysis
    
    def _analyze_edge_attributes(self) -> Dict[str, Any]:
        """Analyze edge attributes with comprehensive coverage"""
        G = self.network_graph
        
        if not G.edges():
            return {}
        
        # Collect ALL attributes from ALL edges
        all_attributes = set()
        edge_data_list = []
        
        for u, v, data in G.edges(data=True):
            edge_data_list.append(data)
            all_attributes.update(data.keys())
        
        if not all_attributes:
            return {'available_attributes': []}
        
        attr_analysis = {'available_attributes': sorted(list(all_attributes))}
        
        print(f"Analyzing {len(edge_data_list)} edges with {len(all_attributes)} attributes")
        
        # Analyze each attribute
        for attr in all_attributes:
            values = []
            edge_count_with_attr = 0
            
            for data in edge_data_list:
                if attr in data:
                    values.append(data[attr])
                    edge_count_with_attr += 1
            
            if not values:
                continue
            
            # Analyze data type distribution
            type_counts = {}
            for val in values:
                val_type = type(val).__name__
                type_counts[val_type] = type_counts.get(val_type, 0) + 1
            
            primary_type = max(type_counts, key=type_counts.get)
            
            if primary_type in ['int', 'float'] and type_counts.get('bool', 0) == 0:
                # Numeric analysis
                numeric_vals = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
                
                if numeric_vals:
                    try:
                        attr_analysis[attr] = {
                            'type': 'numeric',
                            'count': len(values),
                            'edge_coverage': edge_count_with_attr / len(edge_data_list) * 100,
                            'numeric_count': len(numeric_vals),
                            'mean': float(np.mean(numeric_vals)),
                            'min': float(np.min(numeric_vals)),
                            'max': float(np.max(numeric_vals)),
                            'std': float(np.std(numeric_vals))
                        }
                    except Exception:
                        attr_analysis[attr] = {
                            'type': 'numeric',
                            'count': len(values),
                            'edge_coverage': edge_count_with_attr / len(edge_data_list) * 100,
                            'numeric_count': len(numeric_vals),
                            'error': 'Could not calculate statistics',
                            'sample_values': numeric_vals[:3]
                        }
                else:
                    attr_analysis[attr] = {
                        'type': 'numeric',
                        'count': len(values),
                        'edge_coverage': edge_count_with_attr / len(edge_data_list) * 100,
                        'numeric_count': 0,
                        'error': 'No valid numeric values found'
                    }
            elif primary_type == 'LineString':
                # Geometry analysis
                geometries = [v for v in values if hasattr(v, 'coords')]
                if geometries:
                    coord_counts = [len(geom.coords) for geom in geometries]
                    attr_analysis[attr] = {
                        'type': 'geometry',
                        'count': len(values),
                        'edge_coverage': edge_count_with_attr / len(edge_data_list) * 100,
                        'geometry_count': len(geometries),
                        'avg_points': float(np.mean(coord_counts)),
                        'min_points': int(np.min(coord_counts)),
                        'max_points': int(np.max(coord_counts))
                    }
                else:
                    attr_analysis[attr] = {
                        'type': 'geometry',
                        'count': len(values),
                        'edge_coverage': edge_count_with_attr / len(edge_data_list) * 100,
                        'geometry_count': 0,
                        'error': 'No valid geometries found'
                    }
            else:
                # Categorical/mixed analysis
                unique_vals = list(set(str(v) for v in values))
                
                attr_analysis[attr] = {
                    'type': 'categorical',
                    'count': len(values),
                    'edge_coverage': edge_count_with_attr / len(edge_data_list) * 100,
                    'unique': len(unique_vals),
                    'type_distribution': type_counts,
                    'sample_values': unique_vals[:10]
                }
        
        return attr_analysis
        
    def get_node_coordinates(self) -> Dict[int, Tuple[float, float]]:
        """Extract node coordinates for spatial analysis"""
        if self.network_graph is None:
            raise ValueError("Network not loaded.")
        
        coordinates = {}
        for node in self.network_graph.nodes():
            node_data = self.network_graph.nodes[node]
            if 'x' in node_data and 'y' in node_data:
                coordinates[node] = (node_data['x'], node_data['y'])
        
        print(f"Extracted coordinates for {len(coordinates)} nodes")
        return coordinates
    
    def inspect_all_edge_attributes(self):
        """Inspect all edge attributes in detail"""
        G = self.network_graph
        
        if not G.edges():
            print("No edges found")
            return
        
        print(f"Inspecting {G.number_of_edges()} edges...")
        
        all_attrs = set()
        attr_examples = {}
        attr_coverage = {}
        
        for i, (u, v, data) in enumerate(G.edges(data=True)):
            for attr, value in data.items():
                if attr not in all_attrs:
                    all_attrs.add(attr)
                    attr_examples[attr] = value
                    attr_coverage[attr] = 1
                else:
                    attr_coverage[attr] += 1
            
            if i < 3:
                print(f"\nEdge {i+1}: {u} -> {v}")
                print(f"  Attributes: {list(data.keys())}")
                for attr, val in data.items():
                    if isinstance(val, str) and len(val) > 50:
                        print(f"    {attr}: {type(val).__name__} (length: {len(val)})")
                    else:
                        print(f"    {attr}: {val} ({type(val).__name__})")
        
        print(f"\nSummary of all {len(all_attrs)} attributes:")
        for attr in sorted(all_attrs):
            coverage_pct = (attr_coverage[attr] / G.number_of_edges()) * 100
            example = attr_examples[attr]
            example_str = str(example)[:50] + "..." if len(str(example)) > 50 else str(example)
            print(f"  {attr}: {attr_coverage[attr]} edges ({coverage_pct:.1f}%) - Example: {example_str}")
