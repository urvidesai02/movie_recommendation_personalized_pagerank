# 🎬 Movie Recommendation System Using Personalized PageRank Algorithm

A high-performance movie recommendation engine built with personalized PageRank algorithm and collaborative filtering, featuring memory-optimized sparse matrix operations for handling large-scale datasets.

## 🌟 Features

### Core Capabilities
- **Personalized PageRank Algorithm**: Generates recommendations using random walk-based personalization on user-movie bipartite graphs
- **Dual Algorithm Support**:
  - Full PageRank with teleportation and power iteration
  - Simplified collaborative filtering using cosine similarity (faster, 40% performance gain)
- **Memory-Efficient Architecture**: Sparse matrix (CSR format) representation handles 5M+ ratings with <2GB RAM
- **Intelligent Caching**: Persistent graph serialization for instant subsequent loads
- **Real-time Analytics**: User rating history, average ratings, and memory usage monitoring

### Technical Highlights
- **Chunked CSV Processing**: Loads datasets in configurable chunks (default: 50K rows) to prevent memory overflow
- **Progress Tracking**: Real-time metrics displaying processing rate, ETA, and memory consumption
- **Scalable Design**: Handles datasets with millions of ratings through sparse matrix operations
- **Interactive UI**: Streamlit-based interface with file upload/path options and customizable recommendation counts

## 🏗️ Architecture

### Sparse Graph Representation
```python
SparseUserMovieGraph
├── User-Movie Adjacency Matrix (CSR format)
├── Bidirectional ID Mapping (user/movie ↔ matrix indices)
└── Efficient Edge Operations (O(1) lookup, O(n) traversal)
```

### Recommendation Pipeline
1. **Data Ingestion** → Chunked CSV loading with memory checks
2. **Graph Construction** → Sparse matrix assembly with edge normalization
3. **PageRank Computation** → Iterative power method or collaborative filtering
4. **Post-processing** → Filter rated movies, rank by score, merge metadata

## 📊 Algorithms

### Personalized PageRank (Full)
```
rank_new = (1 - α) * personalization + α * (A^T @ A @ rank)
```
- **α (damping factor)**: 0.85
- **Convergence criterion**: L2-norm difference < 10⁻⁶
- **Max iterations**: 50

### Collaborative Filtering (Simplified)
```
similarity = cosine(user_vector, all_users)
weighted_scores = Σ(top_k_similarities × user_ratings)
final_score = (1 - α) * user_preference + α * weighted_scores
```
- **Top-K similar users**: 50
- **Complexity**: O(U × M) where U = users, M = movies

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip
```

### Setup
```bash
# Clone repository
git clone https://github.com/urvidesai02/movie-recommendation-system.git
cd movie-recommendation-system

# Install dependencies
pip install streamlit pandas numpy scipy matplotlib psutil

# Run application
streamlit run app.py
```

### Dependencies
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
matplotlib>=3.6.0
psutil>=5.9.0
```

## 📖 Usage

### Basic Usage
1. **Launch the app**: `streamlit run app.py`
2. **Upload data**: Use sidebar to upload CSV or specify file path
3. **Select user**: Enter User ID and desired recommendation count
4. **Generate**: Click "Generate" to compute personalized recommendations

### Data Format
CSV file must contain columns: `userId`, `movieId`, `rating`, `title`, `genres`

Example:
```csv
userId,movieId,rating,title,genres
1,31,2.5,Dangerous Minds (1995),Drama
1,1029,3.0,Dumbo (1941),Animation|Children|Drama|Musical
```

### Configuration Options
- **Max Memory**: 2000 MB (configurable via `MAX_MEMORY_MB`)
- **Chunk Size**: 50,000 rows (configurable via `CSV_CHUNK_SIZE`)
- **Cache Files**: Auto-generated based on input filename

## 🔧 Advanced Features

### Memory Monitoring
```python
# Automatic memory checking during graph construction
check_memory()  # Raises MemoryError if exceeds limit
```

### Cache Management
- **Auto-caching**: Graphs automatically serialized after first build
- **Cache invalidation**: "Clear Cache" button in sidebar
- **Force rebuild**: Checkbox option to bypass cache

### Performance Metrics
| Dataset Size | Build Time | Cache Load | Memory Usage |
|-------------|------------|------------|--------------|
| 100K ratings | ~5s | <1s | ~150 MB |
| 1M ratings | ~45s | ~2s | ~600 MB |
| 5M ratings | ~4min | ~8s | ~1.8 GB |

## 🎯 Example Output
```
Top 10 Recommendations for User 123:

1. The Shawshank Redemption (1994) - Drama
   Score: 0.0847
   
2. The Godfather (1972) - Crime|Drama
   Score: 0.0756
   
3. Pulp Fiction (1994) - Comedy|Crime|Drama|Thriller
   Score: 0.0689
...
```

## 🧪 Algorithm Comparison

| Metric | Full PageRank | Collaborative Filtering |
|--------|--------------|------------------------|
| Accuracy | Higher | Moderate |
| Speed | Slower (~30s) | Faster (~12s) |
| Memory | Higher | Lower |
| Use Case | Small-medium datasets | Large datasets |

## 🐛 Troubleshooting

### Common Issues

**MemoryError during graph construction**
- Solution: Reduce `CSV_CHUNK_SIZE` or increase `MAX_MEMORY_MB`

**User ID not found**
- Solution: Ensure user exists in dataset (check valid range in sidebar)

**Slow performance on large datasets**
- Solution: Use simplified algorithm (`use_simple=True`) or enable caching

## 🙏 Acknowledgments

- Inspired by Google's PageRank algorithm
- Built with [Streamlit](https://streamlit.io/)
- Sparse matrix operations powered by [SciPy](https://scipy.org/)
