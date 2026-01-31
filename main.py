import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import pickle
import os
import time
import gc
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Title and description
st.title("🎬 Movie Recommendation System")

# ===== CONFIGURATION AND CONSTANTS =====
ALPHA = 0.85
MAX_ITER = 50
GRAPH_CACHE_FILE = "movie_graph_sparse.pkl"
METADATA_CACHE_FILE = "movie_metadata_cache.pkl"
GRAPH_INFO_FILE = "graph_info_cache.pkl"

# Memory limits to prevent crashes
MAX_MEMORY_MB = 2000
CSV_CHUNK_SIZE = 50000
MAX_MOVIES_FOR_FULL_GRAPH = 50000

# ===== MEMORY MONITORING =====
def get_memory_usage_mb():
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except:
        return 0

def check_memory():
    """Check if we're approaching memory limits"""
    mem_mb = get_memory_usage_mb()
    if mem_mb > MAX_MEMORY_MB:
        gc.collect()
        mem_mb = get_memory_usage_mb()
        if mem_mb > MAX_MEMORY_MB:
            raise MemoryError(f"Memory usage ({mem_mb:.0f}MB) exceeds limit ({MAX_MEMORY_MB}MB)")
    return mem_mb

# ===== SPARSE MATRIX BASED GRAPH =====
class SparseUserMovieGraph:
    """Efficient sparse matrix representation of user-movie graph"""
    
    def __init__(self):
        self.user_map = {}  # user_id -> index
        self.movie_map = {}  # movie_id -> index
        self.user_reverse = {}  # index -> user_id
        self.movie_reverse = {}  # index -> movie_id
        self.adjacency_matrix = None
        self.n_users = 0
        self.n_movies = 0
        self.n_edges = 0
    
    def add_edge(self, user_id, movie_id, rating):
        """Add or update edge between user and movie"""
        if user_id not in self.user_map:
            idx = self.n_users
            self.user_map[user_id] = idx
            self.user_reverse[idx] = user_id
            self.n_users += 1
        
        if movie_id not in self.movie_map:
            idx = self.n_movies
            self.movie_map[movie_id] = idx
            self.movie_reverse[idx] = movie_id
            self.n_movies += 1
        
        self.n_edges += 1
    
    def build_matrix(self, edges_list):
        """Build sparse adjacency matrix from edges"""
        if not edges_list:
            self.adjacency_matrix = csr_matrix((self.n_users, self.n_movies))
            return
        
        rows = [self.user_map[u] for u, m, r in edges_list]
        cols = [self.movie_map[m] for u, m, r in edges_list]
        data = [r for u, m, r in edges_list]
        
        self.adjacency_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_movies),
            dtype=np.float32
        )
    
    def get_user_movies(self, user_id):
        """Get movies rated by user"""
        if user_id not in self.user_map:
            return []
        
        user_idx = self.user_map[user_id]
        row = self.adjacency_matrix.getrow(user_idx)
        movie_indices = row.nonzero()[1]
        return [(self.movie_reverse[idx], float(row[0, idx])) for idx in movie_indices]
    
    def personalized_pagerank(self, user_id, alpha=ALPHA, max_iter=MAX_ITER):
        """
        Compute PageRank personalized to user using efficient random walk.
        This avoids creating the massive movie-to-movie matrix.
        """
        if user_id not in self.user_map:
            raise ValueError(f"User {user_id} not found in graph")
        
        user_idx = self.user_map[user_id]
        
        # Get user's rated movies as starting point
        user_row = self.adjacency_matrix.getrow(user_idx)
        user_movie_indices = user_row.nonzero()[1]
        user_ratings = np.array([user_row[0, idx] for idx in user_movie_indices])
        
        if len(user_movie_indices) == 0:
            # User has no ratings, return uniform distribution
            return np.ones(self.n_movies, dtype=np.float32) / self.n_movies
        
        # Normalize user ratings to create personalization vector
        personalization = np.zeros(self.n_movies, dtype=np.float32)
        personalization[user_movie_indices] = user_ratings / user_ratings.sum()
        
        # Normalize adjacency matrix by rows (users)
        A = self.adjacency_matrix.copy().astype(np.float32)
        row_sums = np.array(A.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        row_inv = 1.0 / row_sums
        
        # Create diagonal matrix for row normalization
        D_inv = sp.diags(row_inv, format='csr')
        A_norm = D_inv @ A  # Normalized: each row sums to 1
        
        # Initialize rank vector
        rank = personalization.copy()
        
        # Power iteration using alternating walks
        # Movie -> User -> Movie (through bipartite graph)
        for iteration in range(max_iter):
            # Step 1: Movie scores propagate to users (A_norm^T @ rank)
            # This gives user scores based on their movie preferences
            user_scores = A_norm.T @ rank
            
            # Step 2: User scores propagate back to movies (A_norm @ user_scores)
            # This gives new movie scores
            rank_new = A_norm.T @ user_scores
            
            # Normalize
            rank_sum = rank_new.sum()
            if rank_sum > 0:
                rank_new = rank_new / rank_sum
            
            # Add teleportation (restart probability)
            rank_new = (1 - alpha) * personalization + alpha * rank_new
            
            # Check convergence
            diff = np.linalg.norm(rank_new - rank)
            if diff < 1e-6:
                break
            
            rank = rank_new
        
        return rank
    
    def personalized_pagerank_simple(self, user_id, alpha=0.85, max_iter=30):
        """
        Simpler and faster version using collaborative filtering similarity.
        Finds similar users and aggregates their movie ratings.
        """
        if user_id not in self.user_map:
            raise ValueError(f"User {user_id} not found in graph")
        
        user_idx = self.user_map[user_id]
        
        # Get user's ratings
        user_row = self.adjacency_matrix.getrow(user_idx).toarray().flatten()
        
        if user_row.sum() == 0:
            return np.ones(self.n_movies, dtype=np.float32) / self.n_movies
        
        # Compute similarity with all users using cosine similarity
        # This is much faster than full PageRank
        
        # Normalize user vector
        user_norm = np.linalg.norm(user_row)
        if user_norm == 0:
            user_norm = 1
        
        # Compute similarities in batches to manage memory
        batch_size = 1000
        similarities = np.zeros(self.n_users, dtype=np.float32)
        
        for start_idx in range(0, self.n_users, batch_size):
            end_idx = min(start_idx + batch_size, self.n_users)
            batch = self.adjacency_matrix[start_idx:end_idx].toarray()
            
            # Cosine similarity
            batch_norms = np.linalg.norm(batch, axis=1)
            batch_norms[batch_norms == 0] = 1
            
            batch_sims = (batch @ user_row) / (batch_norms * user_norm)
            similarities[start_idx:end_idx] = batch_sims
        
        # Zero out the user's own similarity
        similarities[user_idx] = 0
        
        # Get top similar users
        top_k = min(50, self.n_users)  # Consider top 50 similar users
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_sims = similarities[top_indices]
        
        # Weight by similarity and aggregate
        weighted_ratings = np.zeros(self.n_movies, dtype=np.float32)
        
        for idx, sim in zip(top_indices, top_sims):
            if sim > 0:
                similar_user_ratings = self.adjacency_matrix.getrow(idx).toarray().flatten()
                weighted_ratings += sim * similar_user_ratings
        
        # Normalize
        if weighted_ratings.sum() > 0:
            weighted_ratings = weighted_ratings / weighted_ratings.sum()
        
        # Blend with user's own preferences
        user_pref = user_row / user_row.sum() if user_row.sum() > 0 else user_row
        scores = (1 - alpha) * user_pref + alpha * weighted_ratings
        
        return scores


def load_csv_chunked(filepath, chunk_size=CSV_CHUNK_SIZE):
    """Load CSV in chunks to manage memory"""
    chunks = []
    try:
        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            chunks.append(chunk)
            check_memory()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        raise
    
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def build_sparse_graph_with_progress(ratings_df):
    """Build sparse matrix graph with progress tracking"""
    st.info(f"🔨 Building sparse graph from {len(ratings_df):,} ratings...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.container()
    
    with metrics_container:
        col1, col2, col3, col4 = st.columns(4)
        metric_processed = col1.empty()
        metric_rate = col2.empty()
        metric_eta = col3.empty()
        metric_memory = col4.empty()
    
    graph = SparseUserMovieGraph()
    edges_list = []
    
    total_rows = len(ratings_df)
    start_time = time.time()
    update_interval = max(1, total_rows // 200)
    
    for idx, row in enumerate(ratings_df.itertuples(index=False)):
        user_id = int(row.userId)
        movie_id = int(row.movieId)
        rating = float(row.rating)
        
        graph.add_edge(user_id, movie_id, rating)
        edges_list.append((user_id, movie_id, rating))
        
        if idx % update_interval == 0 or idx == total_rows - 1:
            progress = (idx + 1) / total_rows
            progress_bar.progress(progress)
            
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total_rows - idx - 1) / rate if rate > 0 else 0
            mem_mb = check_memory()
            
            status_text.markdown(
                f"**Progress:** {idx + 1:,} / {total_rows:,} ({progress * 100:.1f}%) • "
                f"**Elapsed:** {elapsed:.1f}s • **ETA:** {eta:.0f}s"
            )
            
            metric_processed.metric("Processed", f"{idx + 1:,}")
            metric_rate.metric("Rate", f"{rate:,.0f}/sec")
            metric_eta.metric("ETA", f"{eta:.0f}s")
            metric_memory.metric("Memory", f"{mem_mb:.0f}MB")
            
            if idx % (update_interval * 5) == 0:
                gc.collect()
    
    # Build matrix representation
    status_text.text("Building sparse matrix...")
    graph.build_matrix(edges_list)
    
    progress_bar.empty()
    status_text.empty()
    
    elapsed_total = time.time() - start_time
    st.success(f"✅ Graph built in {elapsed_total:.2f}s!")
    st.info(f"📊 {graph.n_users:,} users × {graph.n_movies:,} movies, {graph.n_edges:,} edges")
    
    return graph


def save_sparse_graph(graph, df, graph_file, metadata_file, info_file):
    """Save sparse graph and metadata"""
    try:
        with open(graph_file, 'wb') as f:
            pickle.dump(graph, f)
        
        metadata = df[["movieId", "title", "genres"]].drop_duplicates("movieId")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        info = {
            'n_users': graph.n_users,
            'n_movies': graph.n_movies,
            'n_edges': graph.n_edges,
            'n_unique_users': len(graph.user_map),
            'n_unique_movies': len(graph.movie_map),
            'n_ratings': len(df)
        }
        
        with open(info_file, 'wb') as f:
            pickle.dump(info, f)
        
        return True
    except Exception as e:
        st.error(f"Error saving cache: {e}")
        return False


def load_sparse_graph(graph_file, metadata_file, info_file):
    """Load sparse graph from cache"""
    try:
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f)
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        with open(info_file, 'rb') as f:
            info = pickle.load(f)
        
        return graph, metadata, info
    except Exception as e:
        raise Exception(f"Error loading cache: {e}")


def recommend_for_user(graph, metadata_df, user_id, top_n=10, use_simple=True):
    """Generate recommendations using personalized PageRank"""
    # Use the simpler, faster method by default
    if use_simple:
        pr = graph.personalized_pagerank_simple(user_id)
    else:
        pr = graph.personalized_pagerank(user_id)
    
    # Get already rated movies
    already_rated = set([m for m, r in graph.get_user_movies(user_id)])
    
    # Score all movies
    candidate_scores = []
    for movie_idx, score in enumerate(pr):
        if score > 0:
            movie_id = graph.movie_reverse[movie_idx]
            if movie_id not in already_rated:
                candidate_scores.append((movie_id, float(score)))
    
    # Sort and get top N
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidate_scores[:top_n]
    
    if not top_candidates:
        st.warning("No new movies to recommend")
        return pd.DataFrame(columns=["movieId", "title", "genres", "score"])
    
    rec_df = pd.DataFrame(top_candidates, columns=["movieId", "score"])
    rec_df = rec_df.merge(metadata_df, on="movieId", how="left")
    rec_df = rec_df[["movieId", "title", "genres", "score"]].reset_index(drop=True)
    
    return rec_df


# ===== SIDEBAR CONFIGURATION =====

st.sidebar.header("⚙️ Configuration")

# Tabbed interface for data source
data_source_tab = st.sidebar.radio(
    "Select Data Source:",
    ["📤 Upload File", "📁 File Path"],
    label_visibility="collapsed"
)

DATA_PATH = None

if data_source_tab == "📤 Upload File":
    st.sidebar.subheader("Upload Data File")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Drag and drop or click to upload your ratings CSV file"
    )
    
    if uploaded_file is not None:
        # Create a unique filename
        DATA_PATH = f"./uploaded_{uploaded_file.name}"
        
        # Show file details
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.sidebar.success(f"✅ {uploaded_file.name}")
        st.sidebar.metric("File Size", f"{file_size_mb:.2f} MB")
        
        # Save to disk
        if not os.path.exists(DATA_PATH):
            with st.spinner("💾 Saving uploaded file..."):
                with open(DATA_PATH, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.sidebar.info("File saved successfully")
        
        # Create cache filenames based on uploaded file
        data_basename = uploaded_file.name.replace('.csv', '').replace('.', '_')
        GRAPH_CACHE_FILE = f"cache_{data_basename}_graph.pkl"
        METADATA_CACHE_FILE = f"cache_{data_basename}_metadata.pkl"
        GRAPH_INFO_FILE = f"cache_{data_basename}_info.pkl"
    else:
        st.sidebar.warning("⚠️ No file uploaded")
        st.info("👈 Please upload a CSV file from the sidebar to continue")
        st.stop()

else:  # File Path option
    st.sidebar.subheader("Enter File Path")
    
    DATA_PATH = st.sidebar.text_input(
        "File path:",
        value="./movie.csv",
        placeholder="/path/to/your/data.csv"
    )
    
    if DATA_PATH:
        if os.path.exists(DATA_PATH):
            file_size_mb = os.path.getsize(DATA_PATH) / (1024 * 1024)
            st.sidebar.success("✅ File found")
            st.sidebar.metric("File Size", f"{file_size_mb:.2f} MB")
            
            # Create cache filenames
            data_basename = os.path.basename(DATA_PATH).replace('.csv', '').replace('.', '_')
            GRAPH_CACHE_FILE = f"cache_{data_basename}_graph.pkl"
            METADATA_CACHE_FILE = f"cache_{data_basename}_metadata.pkl"
            GRAPH_INFO_FILE = f"cache_{data_basename}_info.pkl"
        else:
            st.sidebar.error("❌ File not found")
            st.error(f"File not found: {DATA_PATH}")
            st.info("Please check the file path and try again")
            st.stop()
    else:
        st.sidebar.warning("⚠️ Please enter a file path")
        st.stop()

# Check if cache exists for this specific file
cache_exists = (os.path.exists(GRAPH_CACHE_FILE) and 
                os.path.exists(METADATA_CACHE_FILE) and 
                os.path.exists(GRAPH_INFO_FILE))

# Show cache status
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Cache Status")

cache_exists = (os.path.exists(GRAPH_CACHE_FILE) and 
                os.path.exists(METADATA_CACHE_FILE) and 
                os.path.exists(GRAPH_INFO_FILE))

if cache_exists:
    st.sidebar.success("✅ Cached graph found!")
    total_size = sum(os.path.getsize(f) for f in 
                    [GRAPH_CACHE_FILE, METADATA_CACHE_FILE, GRAPH_INFO_FILE]) / (1024 * 1024)
    st.sidebar.metric("Cache Size", f"{total_size:.2f} MB")
else:
    st.sidebar.warning("⚠️ No cached graph found")

force_rebuild = st.sidebar.checkbox("🔄 Force rebuild", value=False)

if st.sidebar.button("🗑️ Clear Cache"):
    for f in [GRAPH_CACHE_FILE, METADATA_CACHE_FILE, GRAPH_INFO_FILE]:
        if os.path.exists(f):
            os.remove(f)
    st.cache_data.clear()
    st.sidebar.success("✅ Cleared!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.write("**Memory Optimization Enabled**")
st.sidebar.info(f"Max memory: {MAX_MEMORY_MB}MB")

# ===== MAIN LOGIC =====
try:
    if cache_exists and not force_rebuild:
        st.info("📂 Loading from cache...")
        load_progress = st.progress(0)
        
        load_progress.progress(33)
        graph, metadata_df, graph_info = load_sparse_graph(
            GRAPH_CACHE_FILE, METADATA_CACHE_FILE, GRAPH_INFO_FILE
        )
        
        load_progress.progress(100)
        load_progress.empty()
        
        st.success("✅ Graph loaded from cache!")
        using_cached_data = True
        full_df = None
        
    else:
        st.info("🔄 Building graph from CSV...")
        full_df = load_csv_chunked(DATA_PATH)
        st.success(f"✅ Loaded {len(full_df):,} ratings")
        
        # Limit graph size if needed
        if len(full_df) > 5_000_000:
            st.warning(f"Dataset is large ({len(full_df):,} rows). This may take several minutes...")
        
        st.markdown("---")
        st.subheader("🔨 Building Graph")
        
        graph = build_sparse_graph_with_progress(full_df)
        metadata_df = full_df[["movieId", "title", "genres"]].drop_duplicates("movieId")
        
        # Save to cache
        st.markdown("---")
        st.info("💾 Saving to cache...")
        if save_sparse_graph(graph, full_df, GRAPH_CACHE_FILE, 
                            METADATA_CACHE_FILE, GRAPH_INFO_FILE):
            st.success("✅ Graph cached! Next load will be instant.")
        
        with open(GRAPH_INFO_FILE, 'rb') as f:
            graph_info = pickle.load(f)
        
        using_cached_data = False
        st.balloons()
    
    # Display stats
    st.sidebar.success("✅ System Ready!")
    st.sidebar.metric("Users", f"{graph_info['n_unique_users']:,}")
    st.sidebar.metric("Movies", f"{graph_info['n_unique_movies']:,}")
    st.sidebar.metric("Ratings", f"{graph_info['n_ratings']:,}")
    
    # ===== RECOMMENDATION INTERFACE =====
    st.markdown("---")
    
    valid_user_ids = sorted(list(graph.user_map.keys()))
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("User Selection")
        user_id = st.number_input(
            "Enter User ID",
            min_value=min(valid_user_ids),
            max_value=max(valid_user_ids),
            value=valid_user_ids[0],
            step=1
        )
        
        top_n = st.slider("Number of recommendations", 5, 50, 10)
        generate_recs = st.button("🎯 Generate", type="primary", use_container_width=True)
    
    with col2:
        st.subheader(f"Results for User {user_id}")
        
        user_movies = graph.get_user_movies(user_id)
        n_rated = len(user_movies)
        
        if n_rated > 0:
            ratings = [r for m, r in user_movies]
            avg_rating = np.mean(ratings)
        else:
            avg_rating = 0
        
        col2_1, col2_2, col2_3 = st.columns(3)
        col2_1.metric("Movies Rated", n_rated)
        col2_2.metric("Avg Rating", f"{avg_rating:.2f}")
        col2_3.metric("Memory Used", f"{get_memory_usage_mb():.0f}MB")
        
        tab1, tab2 = st.tabs(["📊 Recommendations", "📚 History"])
        
        with tab1:
            if generate_recs:
                start_time = time.time()
                with st.spinner("🔮 Generating recommendations..."):
                    rec_df = recommend_for_user(
                        graph, metadata_df, user_id, 
                        top_n=top_n, 
                        use_simple=True
                    )
                    st.session_state['recommendations'] = rec_df
                
                elapsed = time.time() - start_time
                
                if len(rec_df) > 0:
                    st.success(f"✨ Generated {len(rec_df)} recommendations in {elapsed:.2f}s!")
                    
                    for idx, row in rec_df.iterrows():
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.markdown(f"**{idx + 1}. {row['title']}**")
                            st.caption(row['genres'])
                        with col_b:
                            st.metric("Score", f"{row['score']:.4f}")
                        st.divider()
                    
                    csv = rec_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        data=csv,
                        file_name=f"recommendations_user_{user_id}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.info("👆 Click 'Generate' to get recommendations")
        
        with tab2:
            if user_movies:
                history_data = []
                for m_id, rating in sorted(user_movies, key=lambda x: x[1], reverse=True)[:20]:
                    movie_info = metadata_df[metadata_df['movieId'] == m_id]
                    if len(movie_info) > 0:
                        history_data.append({
                            'Movie ID': m_id,
                            'Title': movie_info.iloc[0]['title'],
                            'Genres': movie_info.iloc[0]['genres'],
                            'Rating': rating
                        })
                
                if history_data:
                    st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
                else:
                    st.info("Movie details not available")
            else:
                st.info("No rating history found for this user")

except MemoryError as e:
    st.error(f"❌ {str(e)}")
    st.warning("Try reducing the dataset size or increasing system RAM")
except FileNotFoundError as e:
    st.error(f"❌ File not found: {DATA_PATH}")
    st.info("Please check the file path in the sidebar")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    import traceback
    with st.expander("Debug Info"):
        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    unsafe_allow_html=True
)