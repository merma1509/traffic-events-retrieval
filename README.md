# RoutiQ IR - Traffic Intelligence Search System

## Overview

Information retrieval system for traffic events, enabling search across congestion patterns, weather conditions, and temporal traffic data.

## Project Structure

```bash
routiq-ir/
├── src/
│   ├── data/                # Data loading and processing
│   │   ├── loaders.py       # Traffic & network data ingestion
│   │   ├── generators.py    # Document generation pipeline
│   │   └── preprocessors.py # Text tokenization & cleaning
│   ├── indexing/            # Search indexing
│   │   ├── bm25.py          # BM25 inverted index
│   │   └── indexer.py       # Index building & management
│   ├── retrieval/           # Search functionality
│   │   ├── search.py        # Query processing & ranking
│   │   └── rankers.py       # Scoring algorithms
│   ├── evaluation/          # Performance evaluation
│   │   ├── metrics.py       # MAP, Precision@K, Recall@K
│   │   └── benchmark.py     # Performance evaluation
│   └── ui/                  # User interfaces
│       ├── streamlit_app.py # Web demo interface
│       └── cli.py           # Command-line interface
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Generated corpus
│   └── indices/             # Built search indices
├── tests/                   # Unit tests
├── notebooks/               # Development notebooks
├── requirements.txt         # Python dependencies
├── setup.py                 # Package configuration
└── main.py                  # Entry point & CLI
```

## Core Components

### Data Layer

- Traffic weather data loading
- Kigali network graph processing
- Document generation from traffic events
- Text preprocessing with categorized tokens

### Indexing Layer

- BM25 inverted index implementation
- Efficient index building and persistence

### Retrieval Layer

- Query processing and document ranking
- Multiple scoring algorithms

### Evaluation Layer

- IR evaluation metrics (MAP, P@K, R@K)
- Performance benchmarking

### Interface Layer

- Interactive web demo
- Command-line search tool

## Key Features

- Traffic event search by congestion, weather, temporal patterns
- Spatial queries by location and road type
- Weather filtering (rain, temperature, visibility)
- Temporal search (rush hour, weekend patterns)
- Rich metadata search (vehicle counts, capacity, speed limits)

## Quick Start

```bash
# Build corpus
python main.py build-corpus --size 10000

# Create index
python main.py build-index --corpus processed/traffic_corpus.csv

# Search demo
python main.py search --query "heavy rain congestion residential"

# Launch UI
streamlit run src/ui/streamlit_app.py
```

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```
