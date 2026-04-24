"""Command-line interface for traffic events retrieval system"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

class TrafficSearchCLI:
    """Command-line interface for traffic search system"""
    
    def __init__(self):
        """Initialize CLI"""
        self.commands = {
            'search': self.search,
            'demo': self.demo,
            'help': self.help,
            'exit': self.exit
        }
    
    def search(self, query: str, k: int = 10):
        """Perform search from CLI"""
        try:
            from indexing.bm25_indexer import BM25Indexer
            
            # Create sample index for CLI
            sample_docs = [
                {"doc_id": "1", "text": "Heavy traffic congestion on main road", "all_tokens": ["heavy", "traffic", "congestion", "main", "road"]},
                {"doc_id": "2", "text": "Rain causing poor visibility", "all_tokens": ["rain", "poor", "visibility"]},
                {"doc_id": "3", "text": "Traffic accident blocking lanes", "all_tokens": ["traffic", "accident", "blocking", "lanes"]},
            ]
            
            indexer = BM25Indexer()
            indexer.build_index(sample_docs, "all_tokens")
            
            results = indexer.search(query, k=k)
            
            print(f"\nSearch Results for: '{query}'")
            print(f"Found {len(results)} results\n")
            
            for i, (doc_id, score, doc) in enumerate(results, 1):
                print(f"{i}. {doc['text']}")
                print(f"   Score: {score:.4f}")
                print()
                
        except Exception as e:
            print(f"Search error: {e}")
    
    def demo(self):
        """Run interactive demo"""
        print("RoutiQ IR - CLI Demo")
        print("=" * 30)
        print("Available commands: search, demo, help, exit")
        print()
        
        while True:
            try:
                command = input("routiq> ").strip().lower()
                
                if command == 'exit':
                    break
                elif command == 'help':
                    self.help()
                elif command == 'demo':
                    print("Demo mode already active")
                elif command.startswith('search '):
                    query = command[7:]  # Remove 'search ' prefix
                    self.search(query)
                elif command == 'search':
                    query = input("Enter search query: ").strip()
                    self.search(query)
                else:
                    print(f"Unknown command: {command}")
                    self.help()
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    
    def help(self):
        """Show help information"""
        print("\nRoutiQ IR CLI Help")
        print("=" * 25)
        print("Commands:")
        print("  search <query>  - Search traffic events")
        print("  demo           - Interactive demo mode")
        print("  help           - Show this help")
        print("  exit           - Exit CLI")
        print("\nExamples:")
        print("  search traffic congestion")
        print("  search rain")
        print("  search accident")
        print()
    
    def exit(self):
        """Exit CLI"""
        print("Goodbye!")
        sys.exit(0)
    
    def run(self):
        """Run CLI interface"""
        if len(sys.argv) > 1:
            # Command line mode
            command = sys.argv[1].lower()
            if command in self.commands:
                if command == 'search':
                    query = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else input("Enter search query: ")
                    self.search(query)
                elif command == 'demo':
                    self.demo()
                elif command == 'help':
                    self.help()
                elif command == 'exit':
                    self.exit()
            else:
                print(f"Unknown command: {command}")
                self.help()
        else:
            # Interactive mode
            self.demo()

__all__ = ['TrafficSearchCLI']
