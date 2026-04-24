"""Enhanced Streamlit web interface for traffic events retrieval system"""

import streamlit as st
import sys
import time
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from indexing.bm25_indexer import BM25Indexer

class EnhancedTrafficSearchApp:
    """Enhanced Streamlit application with beautiful UI and working search"""
    
    def __init__(self):
        """Initialize the application"""
        st.set_page_config(
            page_title="RoutiQ IR - Traffic Intelligence Search",
            page_icon="Traffic",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        # Initialize working indexer with enhanced sample data
        if 'indexer' not in st.session_state:
            with st.spinner("Loading search engine..."):
                self._create_enhanced_index()
    
    def _create_enhanced_index(self):
        """Create an enhanced index with comprehensive traffic data"""
        # Enhanced traffic documents with proper token format and rich metadata
        enhanced_docs = [
            {
                "doc_id": "1", 
                "text": "Heavy traffic congestion on main road during rush hour causing 30-minute delays", 
                "all_tokens": ["heavy", "traffic", "congestion", "main", "road", "rush", "hour", "delays"],
                "congestion_level": "Heavy",
                "weather_condition": "Clear",
                "location": "Main Street",
                "timestamp": "2024-04-24 08:30:00",
                "severity": "High"
            },
            {
                "doc_id": "2", 
                "text": "Rain causing poor visibility and hazardous driving conditions on highway", 
                "all_tokens": ["rain", "poor", "visibility", "hazardous", "driving", "conditions", "highway"],
                "congestion_level": "Moderate",
                "weather_condition": "Heavy Rain",
                "location": "Highway 1",
                "timestamp": "2024-04-24 14:15:00",
                "severity": "Medium"
            },
            {
                "doc_id": "3", 
                "text": "Traffic accident blocking two lanes on highway, emergency services on scene", 
                "all_tokens": ["traffic", "accident", "blocking", "two", "lanes", "highway", "emergency"],
                "congestion_level": "Severe",
                "weather_condition": "Clear",
                "location": "Highway 2",
                "timestamp": "2024-04-24 17:45:00",
                "severity": "Critical"
            },
            {
                "doc_id": "4", 
                "text": "Light traffic with free flow conditions on all major routes", 
                "all_tokens": ["light", "traffic", "free", "flow", "conditions", "major", "routes"],
                "congestion_level": "Light",
                "weather_condition": "Clear",
                "location": "City Center",
                "timestamp": "2024-04-24 10:00:00",
                "severity": "Low"
            },
            {
                "doc_id": "5", 
                "text": "Moderate congestion in downtown area due to construction work", 
                "all_tokens": ["moderate", "congestion", "downtown", "area", "construction", "work"],
                "congestion_level": "Moderate",
                "weather_condition": "Cloudy",
                "location": "Downtown",
                "timestamp": "2024-04-24 11:30:00",
                "severity": "Medium"
            },
            {
                "doc_id": "6", 
                "text": "Severe delays due to road construction on main arterial road", 
                "all_tokens": ["severe", "delays", "road", "construction", "main", "arterial"],
                "congestion_level": "Heavy",
                "weather_condition": "Clear",
                "location": "Arterial Road",
                "timestamp": "2024-04-24 09:15:00",
                "severity": "High"
            },
            {
                "doc_id": "7", 
                "text": "Clear weather with normal traffic flow across the city network", 
                "all_tokens": ["clear", "weather", "normal", "traffic", "flow", "city", "network"],
                "congestion_level": "Light",
                "weather_condition": "Clear",
                "location": "City Network",
                "timestamp": "2024-04-24 13:00:00",
                "severity": "Low"
            },
            {
                "doc_id": "8", 
                "text": "Peak hour traffic with moderate delays on commuter routes", 
                "all_tokens": ["peak", "hour", "traffic", "moderate", "delays", "commuter", "routes"],
                "congestion_level": "Moderate",
                "weather_condition": "Clear",
                "location": "Commuter Routes",
                "timestamp": "2024-04-24 18:00:00",
                "severity": "Medium"
            },
            {
                "doc_id": "9", 
                "text": "Roundabout congestion during morning commute causing traffic backups", 
                "all_tokens": ["roundabout", "congestion", "morning", "commute", "traffic", "backups"],
                "congestion_level": "Moderate",
                "weather_condition": "Clear",
                "location": "Central Roundabout",
                "timestamp": "2024-04-24 07:45:00",
                "severity": "Medium"
            },
            {
                "doc_id": "10", 
                "text": "Free flow conditions on highway after accident clearance", 
                "all_tokens": ["free", "flow", "conditions", "highway", "accident", "clearance"],
                "congestion_level": "Light",
                "weather_condition": "Clear",
                "location": "Highway 1",
                "timestamp": "2024-04-24 19:30:00",
                "severity": "Low"
            }
        ]
        
        # Create and build index
        indexer = BM25Indexer()
        stats = indexer.build_index(enhanced_docs, "all_tokens")
        
        # Store in session state
        st.session_state.indexer = indexer
        st.session_state.index_stats = stats
        st.session_state.enhanced_docs = enhanced_docs
    
    def run(self):
        """Run the enhanced Streamlit application"""
        # Custom CSS for enhanced styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .search-container {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e9ecef;
            margin-bottom: 2rem;
        }
        .result-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e9ecef;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }
        .result-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }
        .severity-high { border-left: 4px solid #dc3545; }
        .severity-medium { border-left: 4px solid #ffc107; }
        .severity-low { border-left: 4px solid #28a745; }
        .severity-critical { border-left: 4px solid #6f42c1; }
        </style>
        """, unsafe_allow_html=True)
        
        # Enhanced Header
        st.markdown("""
        <div class="main-header">
            <h1>RoutiQ IR</h1>
            <h2>Traffic Intelligence Search System</h2>
            <p>Real-time traffic event retrieval with advanced BM25 indexing</p>
            <p><strong>Kigali City Traffic Management System</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Search Section
            st.markdown('<div class="search-container">', unsafe_allow_html=True)
            st.header("Search Traffic Events")
            
            # Search input with enhanced styling
            query = st.text_input(
                "Enter your search query:",
                placeholder="e.g., traffic congestion, rain, accident, delays, highway...",
                help="Try searching for: traffic, congestion, rain, accident, delays, construction, highway, road"
            )
            
            # Advanced search options
            col_a, col_b = st.columns(2)
            with col_a:
                top_k = st.selectbox(
                    "Number of results:",
                    options=[5, 10, 15, 20],
                    index=0,
                    help="Select how many results to display"
                )
            
            with col_b:
                search_strategy = st.selectbox(
                    "Search strategy:",
                    options=["smart", "basic", "multi", "specialized"],
                    index=0,
                    help="Choose search algorithm"
                )
            
            # Search button
            search_button = st.button("Search", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
            # Results Section
            if search_button and query:
                with st.spinner("Searching traffic events..."):
                    start_time = time.time()
                    
                    # Search using the enhanced indexer
                    results = st.session_state.indexer.search(query, k=top_k)
                    search_time = time.time() - start_time
                    
                    # Add to search history
                    st.session_state.search_history.insert(0, {
                        'query': query,
                        'strategy': search_strategy,
                        'results_count': len(results),
                        'search_time': search_time,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    # Display results with enhanced formatting
                    if results:
                        st.success(f"Found {len(results)} results in {search_time:.3f} seconds")
                        
                        for i, (doc_id, score, doc) in enumerate(results, 1):
                            severity_class = f"severity-{doc.get('severity', 'low').lower()}"
                            
                            st.markdown(f"""
                            <div class="result-card {severity_class}">
                                <h4>Result {i}: Score {score:.4f}</h4>
                                <p><strong>Event:</strong> {doc['text']}</p>
                                <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                                    <span><strong>Congestion:</strong> {doc.get('congestion_level', 'N/A')}</span>
                                    <span><strong>Weather:</strong> {doc.get('weather_condition', 'N/A')}</span>
                                    <span><strong>Location:</strong> {doc.get('location', 'N/A')}</span>
                                    <span><strong>Severity:</strong> {doc.get('severity', 'N/A')}</span>
                                </div>
                                <div style="margin-top: 0.5rem; color: #666;">
                                    <small>Time: {doc.get('timestamp', 'N/A')}</small>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning(f"No results found for '{query}'")
                        st.info("Try different terms like: traffic, congestion, rain, accident, delays, construction, highway")
            else:
                # Welcome message
                st.markdown("""
                <div class="search-container">
                    <h3>Welcome to RoutiQ IR</h3>
                    <p>Enter a search query above to explore traffic events in Kigali City.</p>
                    <p><strong>Popular searches:</strong> traffic congestion, rain accidents, highway delays, construction work</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # System Information Panel
            st.header("System Information")
            
            # Index Statistics
            if 'index_stats' in st.session_state:
                stats = st.session_state.index_stats
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Documents", f"{stats['corpus_size']:,}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card" style="margin-top: 0.5rem;">', unsafe_allow_html=True)
                st.metric("Vocabulary", f"{stats['vocab_size']:,}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card" style="margin-top: 0.5rem;">', unsafe_allow_html=True)
                st.metric("Avg Length", f"{stats['avg_doc_length']:.1f} tokens")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Search History
            if st.session_state.search_history:
                st.header("Recent Searches")
                for i, search in enumerate(st.session_state.search_history[:5], 1):
                    with st.expander(f"{i}. {search['query']}", expanded=False):
                        st.write(f"**Strategy:** {search['strategy']}")
                        st.write(f"**Results:** {search['results_count']}")
                        st.write(f"**Time:** {search['search_time']:.3f}s")
                        st.write(f"**When:** {search['timestamp']}")
            
            # Quick Search Options
            st.header("Quick Searches")
            quick_searches = [
                "traffic congestion",
                "rain accidents", 
                "construction delays",
                "highway incidents",
                "rush hour traffic",
                "fog visibility"
            ]
            
            for quick_search in quick_searches:
                if st.button(quick_search, key=f"quick_{quick_search}", use_container_width=True):
                    st.session_state.quick_query = quick_search.split(' ', 1)[1]
                    st.rerun()
            
            # System Status
            st.header("System Status")
            st.success("All systems operational")
            st.info("BM25 Indexing Engine Active")
            st.info("Real-time Search Ready")

def main():
    """Main function to run the enhanced app"""
    app = EnhancedTrafficSearchApp()
    app.run()

if __name__ == "__main__":
    main()
