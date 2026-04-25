import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RoutiQVisualizer:
    """Visualization suite for RoutiQ IR system"""
    
    def __init__(self):
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#F4A261',
            'info': '#264653'
        }
    
    def plot_evaluation_metrics(self, metrics: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive evaluation metrics visualization
        
        Args:
            metrics: Dictionary containing evaluation results
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RoutiQ IR - Evaluation Metrics Dashboard', fontsize=16, fontweight='bold')
        
        # Extract metrics for plotting
        k_values = [1, 3, 5, 10]
        precision_scores = [metrics.get(f'avg_precision_at_{k}', 0) for k in k_values]
        recall_scores = [metrics.get(f'avg_recall_at_{k}', 0) for k in k_values]
        ndcg_scores = [metrics.get(f'avg_ndcg_at_{k}', 0) for k in k_values]
        
        # 1. Precision@K and Recall@K
        x = np.arange(len(k_values))
        width = 0.35
        
        ax1.bar(x - width/2, precision_scores, width, label='Precision@K', 
                color=self.colors['primary'], alpha=0.8)
        ax1.bar(x + width/2, recall_scores, width, label='Recall@K', 
                color=self.colors['secondary'], alpha=0.8)
        ax1.set_xlabel('K Value')
        ax1.set_ylabel('Score')
        ax1.set_title('Precision@K vs Recall@K')
        ax1.set_xticks(x)
        ax1.set_xticklabels(k_values)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. NDCG@K
        bars = ax2.bar(k_values, ndcg_scores, color=self.colors['accent'], alpha=0.8)
        ax2.set_xlabel('K Value')
        ax2.set_ylabel('NDCG Score')
        ax2.set_title('NDCG@K Performance')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, ndcg_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 3. Overall Performance Metrics
        metric_names = ['MAP', 'Avg Precision', 'Avg Recall@10', 'Avg NDCG@10']
        metric_values = [
            metrics.get('map_score', 0),
            metrics.get('average_precision', 0),
            metrics.get('avg_recall_at_10', 0),
            metrics.get('avg_ndcg_at_10', 0)
        ]
        
        bars = ax3.bar(metric_names, metric_values, 
                      color=[self.colors['primary'], self.colors['secondary'], 
                            self.colors['accent'], self.colors['success']], alpha=0.8)
        ax3.set_ylabel('Score')
        ax3.set_title('Overall Performance Metrics')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Performance Comparison (Radar Chart)
        categories = ['Precision@5', 'Recall@5', 'Precision@10', 'Recall@10', 'NDCG@10']
        values = [
            metrics.get('avg_precision_at_5', 0),
            metrics.get('avg_recall_at_5', 0),
            metrics.get('avg_precision_at_10', 0),
            metrics.get('avg_recall_at_10', 0),
            metrics.get('avg_ndcg_at_10', 0)
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax4.plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'])
        ax4.fill(angles, values, alpha=0.25, color=self.colors['primary'])
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('Performance Radar Chart')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_search_performance_comparison(self, 
                                         strategies_results: Dict[str, Dict[str, Any]], 
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare performance across different search strategies
        
        Args:
            strategies_results: Results for different search strategies
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        strategies = list(strategies_results.keys())
        metrics = ['precision_at_10', 'recall_at_10', 'ndcg_at_10']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Search Strategy Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Bar chart comparison
        x = np.arange(len(strategies))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [strategies_results[strategy].get(metric, 0) for strategy in strategies]
            ax1.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(),
                   alpha=0.8)
        
        ax1.set_xlabel('Search Strategy')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics by Strategy')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(strategies)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Response time comparison
        response_times = [strategies_results[strategy].get('avg_response_time', 0) 
                         for strategy in strategies]
        
        bars = ax2.bar(strategies, response_times, color=self.colors['accent'], alpha=0.8)
        ax2.set_xlabel('Search Strategy')
        ax2.set_ylabel('Response Time (seconds)')
        ax2.set_title('Average Response Time by Strategy')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time_val in zip(bars, response_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(response_times)*0.01,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_traffic_data_analysis(self, traffic_data: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze and visualize traffic data patterns
        
        Args:
            traffic_data: DataFrame containing traffic data
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Traffic Data Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Highway Type Distribution
        if 'highway_type' in traffic_data.columns:
            highway_counts = traffic_data['highway_type'].value_counts()
            ax1.pie(highway_counts.values, labels=highway_counts.index, autopct='%1.1f%%',
                   colors=[self.colors['primary'], self.colors['secondary'], 
                          self.colors['accent'], self.colors['success']])
            ax1.set_title('Highway Type Distribution')
        
        # 2. Vehicle Counts vs Speed Limit Scatter Plot
        if 'vehicle_counts' in traffic_data.columns and 'speed_limit_kmh' in traffic_data.columns:
            # Calculate average vehicle counts by speed limit
            avg_counts = traffic_data.groupby('speed_limit_kmh')['vehicle_counts'].mean()
            scatter = ax2.scatter(avg_counts.index, avg_counts.values, 
                                alpha=0.6, c=range(len(avg_counts)), 
                                cmap='viridis', s=100)
            ax2.set_xlabel('Speed Limit (km/h)')
            ax2.set_ylabel('Average Vehicle Count')
            ax2.set_title('Vehicle Counts vs Speed Limit')
            ax2.grid(True, alpha=0.3)
        
        # 3. Temporal Patterns - Hourly Vehicle Counts
        if 'hour_of_day' in traffic_data.columns:
            hourly_avg = traffic_data.groupby('hour_of_day')['vehicle_counts'].mean()
            ax3.plot(hourly_avg.index, hourly_avg.values, marker='o', 
                    color=self.colors['primary'], linewidth=2)
            ax3.set_xlabel('Hour of Day')
            ax3.set_ylabel('Average Vehicle Count')
            ax3.set_title('Hourly Traffic Volume Patterns')
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(range(0, 24, 2))
        
        # 4. Weather Impact on Traffic
        if 'is_rain' in traffic_data.columns and 'vehicle_counts' in traffic_data.columns:
            weather_data = traffic_data.copy()
            weather_data['weather_condition'] = weather_data['is_rain'].apply(lambda x: 'Rain' if x else 'Clear')
            weather_counts = weather_data.groupby('weather_condition')['vehicle_counts'].mean()
            bars = ax4.bar(weather_counts.index, weather_counts.values, 
                          color=self.colors['secondary'], alpha=0.8)
            ax4.set_xlabel('Weather Condition')
            ax4.set_ylabel('Average Vehicle Count')
            ax4.set_title('Weather Impact on Traffic Volume')
            
            # Add value labels
            for bar, count in zip(bars, weather_counts.values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(weather_counts.values)*0.01,
                        f'{count:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_index_performance(self, index_stats: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize index performance statistics
        
        Args:
            index_stats: Dictionary containing index statistics
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Index Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Index Size Comparison
        if 'index_sizes' in index_stats:
            indices = list(index_stats['index_sizes'].keys())
            sizes = list(index_stats['index_sizes'].values())
            
            bars = ax1.bar(indices, sizes, color=self.colors['primary'], alpha=0.8)
            ax1.set_ylabel('Index Size (MB)')
            ax1.set_title('Index Size Comparison')
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar, size in zip(bars, sizes):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(sizes)*0.01,
                        f'{size:.1f}', ha='center', va='bottom')
        
        # 2. Build Time Performance
        if 'build_times' in index_stats:
            indices = list(index_stats['build_times'].keys())
            times = list(index_stats['build_times'].values())
            
            bars = ax2.bar(indices, times, color=self.colors['secondary'], alpha=0.8)
            ax2.set_ylabel('Build Time (seconds)')
            ax2.set_title('Index Build Time')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                        f'{time_val:.1f}s', ha='center', va='bottom')
        
        # 3. Query Performance by Index
        if 'query_performance' in index_stats:
            indices = list(index_stats['query_performance'].keys())
            query_times = list(index_stats['query_performance'].values())
            
            bars = ax3.bar(indices, query_times, color=self.colors['accent'], alpha=0.8)
            ax3.set_ylabel('Average Query Time (ms)')
            ax3.set_title('Query Performance by Index')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for bar, qtime in zip(bars, query_times):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(query_times)*0.01,
                        f'{qtime:.1f}', ha='center', va='bottom')
        
        # 4. Memory Usage
        if 'memory_usage' in index_stats:
            components = list(index_stats['memory_usage'].keys())
            memory = list(index_stats['memory_usage'].values())
            
            ax4.pie(memory, labels=components, autopct='%1.1f%%',
                   colors=[self.colors['primary'], self.colors['secondary'], 
                          self.colors['accent'], self.colors['success']])
            ax4.set_title('Memory Usage Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, evaluation_data: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive dashboard using Plotly
        
        Args:
            evaluation_data: Dictionary containing evaluation results
            save_path: Path to save the HTML file
            
        Returns:
            Plotly Figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Precision@K vs Recall@K', 'NDCG@K Performance', 
                          'Overall Metrics', 'Performance Radar'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "polar"}]]
        )
        
        # Extract data
        k_values = [1, 3, 5, 10]
        precision_scores = [evaluation_data.get(f'avg_precision_at_{k}', 0) for k in k_values]
        recall_scores = [evaluation_data.get(f'avg_recall_at_{k}', 0) for k in k_values]
        ndcg_scores = [evaluation_data.get(f'avg_ndcg_at_{k}', 0) for k in k_values]
        
        # 1. Precision@K vs Recall@K
        fig.add_trace(
            go.Bar(name='Precision@K', x=k_values, y=precision_scores, 
                  marker_color='lightblue', opacity=0.8),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Recall@K', x=k_values, y=recall_scores, 
                  marker_color='lightcoral', opacity=0.8),
            row=1, col=1
        )
        
        # 2. NDCG@K
        fig.add_trace(
            go.Bar(name='NDCG@K', x=k_values, y=ndcg_scores, 
                  marker_color='lightgreen', opacity=0.8),
            row=1, col=2
        )
        
        # 3. Overall Metrics
        metric_names = ['MAP', 'Avg Precision', 'Avg Recall@10', 'Avg NDCG@10']
        metric_values = [
            evaluation_data.get('map_score', 0),
            evaluation_data.get('average_precision', 0),
            evaluation_data.get('avg_recall_at_10', 0),
            evaluation_data.get('avg_ndcg_at_10', 0)
        ]
        
        fig.add_trace(
            go.Bar(name='Overall Metrics', x=metric_names, y=metric_values, 
                  marker_color='gold', opacity=0.8),
            row=2, col=1
        )
        
        # 4. Radar Chart
        categories = ['Precision@5', 'Recall@5', 'Precision@10', 'Recall@10', 'NDCG@10']
        values = [
            evaluation_data.get('avg_precision_at_5', 0),
            evaluation_data.get('avg_recall_at_5', 0),
            evaluation_data.get('avg_precision_at_10', 0),
            evaluation_data.get('avg_recall_at_10', 0),
            evaluation_data.get('avg_ndcg_at_10', 0)
        ]
        
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=categories + categories[:1],
                fill='toself',
                name='Performance',
                line_color='blue',
                fillcolor='rgba(0,100,255,0.25)'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="RoutiQ IR - Interactive Evaluation Dashboard",
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_query_analysis(self, query_results: List[Dict[str, Any]], save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze and visualize query performance patterns
        
        Args:
            query_results: List of query result dictionaries
            save_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Query Performance Analysis', fontsize=16, fontweight='bold')
        
        # Extract query data
        query_lengths = [len(result.get('query', '').split()) for result in query_results]
        precision_scores = [result.get('precision_at_10', 0) for result in query_results]
        recall_scores = [result.get('recall_at_10', 0) for result in query_results]
        response_times = [result.get('response_time', 0) for result in query_results]
        
        # 1. Query Length vs Performance
        scatter = ax1.scatter(query_lengths, precision_scores, alpha=0.6, 
                            c=recall_scores, cmap='viridis')
        ax1.set_xlabel('Query Length (words)')
        ax1.set_ylabel('Precision@10')
        ax1.set_title('Query Length vs Precision')
        plt.colorbar(scatter, ax=ax1, label='Recall@10')
        
        # 2. Response Time Distribution
        ax2.hist(response_times, bins=20, color=self.colors['primary'], alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Response Time (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Response Time Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance Distribution
        performance_data = [precision_scores, recall_scores]
        labels = ['Precision@10', 'Recall@10']
        
        bp = ax3.boxplot(performance_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], [self.colors['secondary'], self.colors['accent']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_ylabel('Score')
        ax3.set_title('Performance Distribution')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. Query Success Rate
        success_threshold = 0.5
        successful_queries = sum(1 for p in precision_scores if p >= success_threshold)
        total_queries = len(precision_scores)
        success_rate = successful_queries / total_queries if total_queries > 0 else 0
        
        sizes = [success_rate, 1 - success_rate]
        labels = ['Successful', 'Unsuccessful']
        colors = [self.colors['success'], self.colors['warning']]
        
        ax4.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
        ax4.set_title(f'Query Success Rate (Threshold: {success_threshold})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

# Function for demonstration
def create_sample_visualizations():
    """Create sample visualizations for demonstration purposes"""
    visualizer = RoutiQVisualizer()
    
    # Sample evaluation metrics
    sample_metrics = {
        'total_queries': 100,
        'map_score': 0.75,
        'average_precision': 0.78,
        'avg_precision_at_1': 0.82,
        'avg_precision_at_3': 0.80,
        'avg_precision_at_5': 0.79,
        'avg_precision_at_10': 0.75,
        'avg_recall_at_1': 0.35,
        'avg_recall_at_3': 0.58,
        'avg_recall_at_5': 0.68,
        'avg_recall_at_10': 0.78,
        'avg_ndcg_at_1': 0.82,
        'avg_ndcg_at_3': 0.80,
        'avg_ndcg_at_5': 0.79,
        'avg_ndcg_at_10': 0.81
    }
    
    # Create evaluation metrics plot
    fig1 = visualizer.plot_evaluation_metrics(sample_metrics, 'evaluation_metrics.png')
    
    # Sample strategy comparison
    strategy_results = {
        'Smart': {'precision_at_10': 0.82, 'recall_at_10': 0.78, 'ndcg_at_10': 0.81, 'avg_response_time': 0.15},
        'Multi': {'precision_at_10': 0.79, 'recall_at_10': 0.82, 'ndcg_at_10': 0.80, 'avg_response_time': 0.25},
        'Specialized': {'precision_at_10': 0.85, 'recall_at_10': 0.72, 'ndcg_at_10': 0.83, 'avg_response_time': 0.12},
        'Basic': {'precision_at_10': 0.75, 'recall_at_10': 0.75, 'ndcg_at_10': 0.76, 'avg_response_time': 0.08}
    }
    
    fig2 = visualizer.plot_search_performance_comparison(strategy_results, 'strategy_comparison.png')
    
    # Create interactive dashboard
    interactive_fig = visualizer.create_interactive_dashboard(sample_metrics, 'dashboard.html')
    
    print("Sample visualizations created successfully!")
    print("- evaluation_metrics.png")
    print("- strategy_comparison.png") 
    print("- dashboard.html")

if __name__ == "__main__":
    create_sample_visualizations()
