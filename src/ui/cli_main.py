#!/usr/bin/env python3
"""CLI entry point for RoutiQ IR traffic search system"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

if __name__ == "__main__":
    from cli import TrafficSearchCLI
    
    print("Starting RoutiQ IR CLI...")
    cli = TrafficSearchCLI()
    cli.run()
