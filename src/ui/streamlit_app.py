"""Optimized Streamlit web interface with compact, attractive layout"""

import streamlit as st
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from indexing.bm25_indexer import BM25Indexer

class OptimizedTrafficSearchApp:
    """Optimized Streamlit application with compact, attractive layout"""
    
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
        enhanced_docs = [
            {
                "doc_id": "1", 
                "text": "Heavy traffic congestion on main road during rush hour causing 30-minute delays", 
                "all_tokens": ["heavy", "traffic", "congestion", "main", "road", "rush", "hour", "delays"],
                "congestion_level": "Heavy",
                "weather_condition": "Clear",
                "location": "Main Street",
                "timestamp": "2024-04-24 08:30:00",
                "severity": "High",
                "impact_score": 8.5
            },
            {
                "doc_id": "2", 
                "text": "Rain causing poor visibility and hazardous driving conditions on highway", 
                "all_tokens": ["rain", "poor", "visibility", "hazardous", "driving", "conditions", "highway"],
                "congestion_level": "Moderate",
                "weather_condition": "Heavy Rain",
                "location": "Highway 1",
                "timestamp": "2024-04-24 14:15:00",
                "severity": "Medium",
                "impact_score": 6.2
            },
            {
                "doc_id": "3", 
                "text": "Traffic accident blocking two lanes on highway, emergency services on scene", 
                "all_tokens": ["traffic", "accident", "blocking", "two", "lanes", "highway", "emergency"],
                "congestion_level": "Severe",
                "weather_condition": "Clear",
                "location": "Highway 2",
                "timestamp": "2024-04-24 17:45:00",
                "severity": "Critical",
                "impact_score": 9.8
            },
            {
                "doc_id": "4", 
                "text": "Light traffic with free flow conditions on all major routes", 
                "all_tokens": ["light", "traffic", "free", "flow", "conditions", "major", "routes"],
                "congestion_level": "Light",
                "weather_condition": "Clear",
                "location": "City Center",
                "timestamp": "2024-04-24 10:00:00",
                "severity": "Low",
                "impact_score": 2.1
            },
            {
                "doc_id": "5", 
                "text": "Moderate congestion in downtown area due to construction work", 
                "all_tokens": ["moderate", "congestion", "downtown", "area", "construction", "work"],
                "congestion_level": "Moderate",
                "weather_condition": "Cloudy",
                "location": "Downtown",
                "timestamp": "2024-04-24 11:30:00",
                "severity": "Medium",
                "impact_score": 5.5
            }
        ]
        
        # Create and build index
        indexer = BM25Indexer()
        stats = indexer.build_index(enhanced_docs, "all_tokens")
        
        # Store in session state
        st.session_state.indexer = indexer
        st.session_state.index_stats = stats
        st.session_state.enhanced_docs = enhanced_docs
    
    def _get_severity_color(self, severity):
        """Get color based on severity level"""
        colors = {
            'Low': '#10b981',
            'Medium': '#f59e0b', 
            'High': '#ef4444',
            'Critical': '#8b5cf6'
        }
        return colors.get(severity, '#6b7280')
    
    def run(self):
        """Run the optimized Streamlit application"""
        # Compact CSS with no scrolling
        st.markdown("""
        <style>
        /* Compact Layout - No Scrolling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
        }
        
        .main-header h2 {
            font-size: 1.2rem;
            font-weight: 300;
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
        }
        
        /* Compact Search Container */
        .search-container {
            background: #f8fafc;
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        /* Compact Result Cards */
        .result-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            margin-bottom: 0.8rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        
        .result-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 3px;
            height: 100%;
            background: var(--severity-color, #6b7280);
        }
        
        .result-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .result-card h4 {
            color: #1e293b;
            font-size: 1rem;
            font-weight: 600;
            margin: 0 0 0.5rem 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .result-card .event-text {
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.4;
            margin: 0 0 0.8rem 0;
            padding: 0.5rem;
            background: #f8fafc;
            border-radius: 6px;
            border-left: 2px solid #cbd5e1;
        }
        
        .result-card .metadata-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            margin-bottom: 0.8rem;
        }
        
        .result-card .metadata-item {
            display: flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.3rem;
            background: #f1f5f9;
            border-radius: 6px;
            font-size: 0.8rem;
        }
        
        .result-card .metadata-item strong {
            color: #334155;
            font-weight: 600;
        }
        
        .result-card .impact-bar {
            height: 4px;
            background: #e2e8f0;
            border-radius: 2px;
            overflow: hidden;
        }
        
        .result-card .impact-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%);
            border-radius: 2px;
            transition: width 0.5s ease;
        }
        
        /* Compact Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
            margin-bottom: 0.8rem;
        }
        
        .metric-card .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        
        .metric-card .metric-label {
            font-size: 0.8rem;
            opacity: 0.9;
        }
        
        /* Severity Colors */
        .severity-low { --severity-color: #10b981; }
        .severity-medium { --severity-color: #f59e0b; }
        .severity-high { --severity-color: #ef4444; }
        .severity-critical { --severity-color: #8b5cf6; }
        
        /* Status Indicators */
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.75rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        
        .status-online {
            background: #10b981;
            color: white;
        }
        
        .status-processing {
            background: #f59e0b;
            color: white;
        }
        
        /* Quick Search Buttons */
        .quick-search-btn {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            padding: 0.5rem 0.8rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 500;
            color: #334155;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: center;
            margin-bottom: 0.3rem;
        }
        
        .quick-search-btn:hover {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }
        
        /* Hide streamlit footer and extra padding */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        .stDeployButton {
            display: none;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .result-card .metadata-grid {
                grid-template-columns: 1fr;
            }
            
            .main-header h1 {
                font-size: 1.5rem;
            }
            
            .main-header h2 {
                font-size: 1rem;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Compact Header
        st.markdown("""
        <div class="main-header">
            <h1>RoutiQ IR</h1>
            <h2>Traffic Intelligence Search System</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Main layout with optimized columns
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Compact Search Section
            st.markdown('<div class="search-container">', unsafe_allow_html=True)
            
            # Search input with compact layout
            col_search1, col_search2, col_search3 = st.columns([1, 2, 1])
            with col_search2:
                query = st.text_input(
                    "Search traffic events:",
                    placeholder="e.g., traffic congestion, rain, accident...",
                    help="Enter keywords to search traffic events",
                    label_visibility="visible"
                )
            
            # Compact search options
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            with col_opt1:
                top_k = st.selectbox(
                    "Results",
                    options=[5, 10, 15, 20],
                    index=0,
                    help="Number of results"
                )
            
            with col_opt2:
                search_strategy = st.selectbox(
                    "Strategy",
                    options=["smart", "basic", "multi", "specialized"],
                    index=0,
                    help="Search algorithm"
                )
            
            with col_opt3:
                sort_by = st.selectbox(
                    "Sort by",
                    options=["relevance", "severity", "time"],
                    index=0,
                    help="Sort method"
                )
            
            # Compact search button
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                search_button = st.button(
                    "Search Traffic Events", 
                    type="primary", 
                    use_container_width=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
            # Compact Results Section
            if search_button and query:
                with st.spinner("Searching..."):
                    start_time = time.time()
                    results = st.session_state.indexer.search(query, k=top_k)
                    search_time = time.time() - start_time
                    
                    # Add to search history
                    st.session_state.search_history.insert(0, {
                        'query': query,
                        'strategy': search_strategy,
                        'results_count': len(results),
                        'search_time': search_time,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'sort_by': sort_by
                    })
                    
                    # Display results with compact formatting
                    if results:
                        # Compact success message
                        st.success(f"Found {len(results)} results in {search_time:.3f}s")
                        
                        # Compact result cards
                        for i, (doc_id, score, doc) in enumerate(results, 1):
                            severity_class = f"severity-{doc.get('severity', 'low').lower()}"
                            severity_color = self._get_severity_color(doc.get('severity', 'low'))
                            impact_score = doc.get('impact_score', 5.0)
                            
                            st.markdown(f"""
                            <div class="result-card {severity_class}" style="--severity-color: {severity_color};">
                                <h4>
                                    Result {i}
                                    <span style="font-size: 0.8rem; color: #64748b;">Score: {score:.4f}</span>
                                </h4>
                                <div class="event-text">
                                    {doc['text']}
                                </div>
                                <div class="metadata-grid">
                                    <div class="metadata-item">
                                        <strong>Congestion:</strong> {doc.get('congestion_level', 'N/A')}
                                    </div>
                                    <div class="metadata-item">
                                        <strong>Weather:</strong> {doc.get('weather_condition', 'N/A')}
                                    </div>
                                    <div class="metadata-item">
                                        <strong>Location:</strong> {doc.get('location', 'N/A')}
                                    </div>
                                    <div class="metadata-item">
                                        <strong>Severity:</strong> {doc.get('severity', 'N/A')}
                                    </div>
                                    <div class="metadata-item">
                                        <strong>Time:</strong> {doc.get('timestamp', 'N/A')}
                                    </div>
                                    <div class="metadata-item">
                                        <strong>Impact:</strong> {impact_score:.1f}/10
                                    </div>
                                </div>
                                <div class="impact-bar">
                                    <div class="impact-fill" style="width: {impact_score * 10}%"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Compact no results message
                        st.warning(f"No results found for '{query}'")
                        st.info("Try: traffic, congestion, rain, accident, delays, construction, highway")
            else:
                # Compact welcome message
                st.markdown("""
                <div class="search-container">
                    <h4 style="color: #1e293b; margin-bottom: 0.5rem;">Welcome to RoutiQ IR</h4>
                    <p style="color: #64748b; margin-bottom: 0.5rem;">
                        Enter a search query to explore traffic events in Kigali City.
                    </p>
                    <div style="background: #f1f5f9; padding: 0.5rem; border-radius: 6px; border-left: 3px solid #667eea; font-size: 0.85rem;">
                        <strong style="color: #334155;">Popular searches:</strong> 
                        <span style="color: #64748b;">traffic congestion, rain accidents, highway delays, construction work</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Compact System Information Panel
            st.markdown('<h4 style="color: #1e293b; margin-bottom: 0.8rem;">System Information</h4>', unsafe_allow_html=True)
            
            # Compact Index Statistics
            if 'index_stats' in st.session_state:
                stats = st.session_state.index_stats
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{stats["corpus_size"]:,}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Documents</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{stats["vocab_size"]:,}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Vocabulary</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{stats["avg_doc_length"]:.1f}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Avg Length</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Compact Search History
            if st.session_state.search_history:
                st.markdown('<h4 style="color: #1e293b; margin: 1rem 0 0.8rem 0;">Recent Searches</h4>', unsafe_allow_html=True)
                
                for i, search in enumerate(st.session_state.search_history[:3], 1):
                    with st.expander(f"{i}. {search['query']}", expanded=False):
                        st.markdown(f"""
                        <div style="font-size: 0.85rem; line-height: 1.4;">
                            <div><strong>Strategy:</strong> {search['strategy']}</div>
                            <div><strong>Results:</strong> {search['results_count']}</div>
                            <div><strong>Time:</strong> {search['search_time']:.3f}s</div>
                            <div><strong>When:</strong> {search['timestamp']}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Compact Quick Search Options
            st.markdown('<h4 style="color: #1e293b; margin: 1rem 0 0.8rem 0;">Quick Searches</h4>', unsafe_allow_html=True)
            
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
                    st.session_state.quick_query = quick_search
                    st.rerun()
            
            # Compact System Status
            st.markdown('<h4 style="color: #1e293b; margin: 1rem 0 0.8rem 0;">System Status</h4>', unsafe_allow_html=True)
            
            st.markdown('<div class="status-indicator status-online">All Systems Operational</div>', unsafe_allow_html=True)
            st.markdown('<div class="status-indicator status-processing">BM25 Indexing Active</div>', unsafe_allow_html=True)
            st.markdown('<div class="status-indicator status-online">Real-time Search Ready</div>', unsafe_allow_html=True)
            
            # Compact Performance Metrics
            st.markdown('<h4 style="color: #1e293b; margin: 1rem 0 0.8rem 0;">Performance</h4>', unsafe_allow_html=True)
            
            if st.session_state.search_history:
                avg_time = sum(s['search_time'] for s in st.session_state.search_history) / len(st.session_state.search_history)
                total_searches = len(st.session_state.search_history)
                
                st.metric("Avg Search Time", f"{avg_time:.3f}s")
                st.metric("Total Searches", f"{total_searches}")
                st.metric("Success Rate", "100%")

def main():
    """Main function to run the optimized app"""
    app = OptimizedTrafficSearchApp()
    app.run()

if __name__ == "__main__":
    main()
