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
            from retrieval.retrieval_engine import RetrievalEngine
            
            # Use production retrieval engine with existing indices
            retrieval_engine = RetrievalEngine(indices_dir="data/indices")
            
            results = retrieval_engine.search(query, k=k, strategy="smart")
            search_results = results.get('results', [])
            
            print(f"\nSearch Results for: '{query}'")
            print(f"Found {len(search_results)} results\n")
            
            for i, result in enumerate(search_results, 1):
                doc = result.get('document', {})
                print(f"{i}. {doc.get('text', 'N/A')}")
                print(f"   Score: {result.get('score', 0):.4f}")
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
