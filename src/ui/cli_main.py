#!/usr/bin/env python3
"""CLI entry point for RoutiQ IR traffic search system"""

if __name__ == "__main__":
    from cli import TrafficSearchCLI
    
    cli = TrafficSearchCLI()
    cli.run()
