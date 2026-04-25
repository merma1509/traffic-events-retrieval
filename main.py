import click
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from data.traffic_loader import TrafficWeatherDataLoader
from data.document_generator import TrafficEventDocumentGenerator
from indexing.index_manager import IndexManager
from retrieval.retrieval_engine import RetrievalEngine
from evaluation.evaluator import EvaluationFramework


@click.group()
def cli():
    """RoutiQ IR - Traffic Intelligence Search System"""
    pass


@cli.command()
@click.option('--size', default=10000, help='Number of documents to generate')
@click.option('--output', default='./processed', help='Output directory')
def build_corpus(size, output):
    """Build traffic event corpus from raw data"""
    print(f"Building corpus with {size} documents...")
    print(f"Output directory: {output}")
    
    # Create output directory
    Path(output).mkdir(parents=True, exist_ok=True)
    
    # Load traffic data
    loader = TrafficWeatherDataLoader()
    loader.load_data('data/raw/traffic_weather_temporal.csv')
    
    # Generate documents
    doc_generator = TrafficEventDocumentGenerator(loader.traffic_data)
    corpus = doc_generator.generate_corpus(size)
    
    # Save corpus
    from data.corpus_saver import CorpusSaver
    saver = CorpusSaver()
    saver.save_corpus(corpus, output)
    
    print("Corpus building complete!")


@cli.command()
@click.option('--corpus', required=True, help='Path to corpus file (CSV or PKL)')
@click.option('--output', default='./indices', help='Output directory for index')
def build_index(corpus, output):
    """Build search index from corpus"""
    print(f"Building index from: {corpus}")
    print(f"Output directory: {output}")
    
    # Create output directory
    Path(output).mkdir(parents=True, exist_ok=True)
    
    # Initialize index manager
    index_manager = IndexManager()
    
    # Load corpus
    if corpus.endswith('.pkl'):
        import pickle
        with open(corpus, 'rb') as f:
            corpus = pickle.load(f)
    else:
        import pandas as pd
        corpus = pd.read_csv(corpus).to_dict('records')
    
    # Build main index
    index_manager.create_index(corpus, 'main', 'all_tokens')
    
    # Build specialized indices
    congestion_docs = [doc for doc in corpus if doc.get('congestion_level')]
    weather_docs = [doc for doc in corpus if doc.get('weather_condition')]
    spatial_docs = [doc for doc in corpus if doc.get('spatial_tokens')]
    temporal_docs = [doc for doc in corpus if doc.get('temporal_tokens')]
    
    index_manager.create_index(congestion_docs, 'congestion', 'all_tokens')
    index_manager.create_index(weather_docs, 'weather', 'all_tokens')
    index_manager.create_index(spatial_docs, 'spatial', 'spatial_tokens')
    index_manager.create_index(temporal_docs, 'temporal', 'temporal_tokens')
    
    print("Index building complete!")


@cli.command()
@click.option('--query', required=True, help='Search query')
@click.option('--index', default='data/indices', help='Index directory')
@click.option('--k', default=10, help='Number of results to return')
@click.option('--strategy', default='smart', help='Search strategy (smart, multi, basic, specialized)')
def search(query, index, k, strategy):
    """Search traffic events"""
    print(f"Searching for: '{query}'")
    print(f"Index: {index}")
    print(f"K: {k}")
    print(f"Strategy: {strategy}")
    
    # Initialize retrieval engine
    retrieval_engine = RetrievalEngine(indices_dir=index)
    
    # Perform search
    results = retrieval_engine.search(query, k=k, strategy=strategy)
    
    # Display results
    search_results = results.get('results', [])
    print(f"\nFound {len(search_results)} results:")
    for i, result in enumerate(search_results, 1):
        print(f"\n{i}. {result.get('doc_id', 'N/A')}")
        print(f"   Score: {result.get('score', 0):.4f}")
        if 'document' in result:
            doc = result['document']
            print(f"   Text: {doc.get('text', 'N/A')[:100]}...")
            print(f"   Congestion: {doc.get('congestion_level', 'N/A')}")
            print(f"   Weather: {doc.get('weather_condition', 'N/A')}")
        else:
            print(f"   Data: {result}")
    
    # Show search metadata
    metadata = results.get('metadata', {})
    if metadata:
        print(f"\nSearch time: {metadata.get('search_time', 0):.3f}s")
        print(f"Strategy used: {metadata.get('strategy_used', 'N/A')}")
    
    print("\nSearch complete!")


@cli.command()
@click.option('--corpus', default='data/processed/traffic_corpus_20260424_143258.csv', help='Path to corpus file (CSV or PKL)')
@click.option('--queries', help='Path to queries file (JSON format)')
@click.option('--output', default='./evaluation_results.json', help='Output file for results')
@click.option('--k-values', default='1,3,5,10', help='K values for evaluation (comma-separated)')
def evaluate(corpus, queries, output, k_values):
    """Evaluate retrieval system"""
    print(f"Evaluating system with corpus: {corpus}")
    
    # Parse k values
    k_values = [int(k.strip()) for k in k_values.split(',')]
    
    # Initialize evaluation framework
    evaluator = EvaluationFramework()
    
    # Load corpus
    if corpus.endswith('.pkl'):
        import pickle
        with open(corpus, 'rb') as f:
            corpus = pickle.load(f)
    else:
        import pandas as pd
        corpus = pd.read_csv(corpus).to_dict('records')
    
    # Use sample queries if none provided
    if not queries:
        print("No queries file provided, using sample queries...")
        queries = [
            "heavy congestion",
            "rain during rush hour", 
            "traffic accident",
            "clear weather conditions",
            "moderate traffic flow"
        ]
        
        # Create sample qrels
        qrels = evaluator.create_sample_qrels(corpus, queries)
    else:
        # Load queries and qrels from file
        import json
        with open(queries, 'r') as f:
            queries_data = json.load(f)
            queries = queries_data['queries']
            qrels = queries_data['qrels']
    
    print(f"Using {len(queries)} queries")
    print(f"Evaluating with k values: {k_values}")
    
    # Run evaluation
    results = evaluator.evaluate_system(queries, qrels, k_values, output)
    
    print("Evaluation complete!")


@cli.command()
@click.option('--index', default='data/indices', help='Index directory')
def demo(index):
    """Interactive demo of the search system"""
    print("RoutiQ IR - Interactive Demo")
    print("=" * 50)
    
    # Initialize retrieval engine
    retrieval_engine = RetrievalEngine(indices_dir=index)
    
    print("Available indices:", retrieval_engine.index_manager.list_indices())
    print("Available strategies: smart, multi, basic, specialized")
    print("\nEnter 'quit' to exit")
    
    while True:
        try:
            query = input("\nEnter search query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Get strategy
            strategy = input("Strategy (smart/multi/basic/specialized) [smart]: ").strip() or 'smart'
            if strategy not in ['smart', 'multi', 'basic', 'specialized']:
                strategy = 'smart'
            
            # Get top_k
            try:
                top_k = int(input("Number of results [10]: ").strip() or '10')
            except ValueError:
                top_k = 10
            
            # Perform search
            results = retrieval_engine.search(query, k=top_k, strategy=strategy)
            
            # Display results
            search_results = results.get('results', [])
            print(f"\nFound {len(search_results)} results:")
            for i, result in enumerate(search_results, 1):
                print(f"\n{i}. {result.get('doc_id', 'N/A')}")
                print(f"   Score: {result.get('score', 0):.4f}")
                if 'document' in result:
                    doc = result['document']
                    print(f"   Text: {doc.get('text', 'N/A')[:100]}...")
                    print(f"   Congestion: {doc.get('congestion_level', 'N/A')}")
                    print(f"   Weather: {doc.get('weather_condition', 'N/A')}")
                else:
                    print(f"   Data: {result}")
            
            # Show search metadata
            metadata = results.get('metadata', {})
            if metadata:
                print(f"\nSearch time: {metadata.get('search_time', 0):.3f}s")
                print(f"Strategy used: {metadata.get('strategy_used', 'N/A')}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point"""
    cli()


if __name__ == '__main__':
    main()
