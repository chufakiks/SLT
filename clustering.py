import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from collections import Counter

embeddings = np.load('video_embeddings.npy')

def compute_similarity_matrix(embeddings):
    """
    Compute pairwise cosine similarity matrix.
    LVFace embeddings are already L2-normalized.
    """
    # For normalized vectors: cosine_sim = dot product
    similarity_matrix = embeddings @ embeddings.T
    return similarity_matrix


def cluster_hierarchical(embeddings, threshold=0.65, linkage='average'):
    """
    Hierarchical clustering using verification threshold.
    
    Args:
        embeddings: (N, 512) normalized face embeddings
        threshold: Similarity threshold for same person (0.65 from paper)
        linkage: 'average', 'complete', or 'single'
    
    Returns:
        labels: Cluster assignment for each face
    """
    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(embeddings)
    
    # Convert similarity to distance
    distance_matrix = 1 - sim_matrix
    
    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,  # 1 - 0.65 = 0.35 distance
        metric='precomputed',
        linkage=linkage
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    return labels, sim_matrix


def cluster_dbscan(embeddings, eps=0.35, min_samples=2):
    """
    DBSCAN clustering - good for finding outliers.
    
    Args:
        embeddings: (N, 512) normalized face embeddings
        eps: Maximum distance between samples (1 - similarity_threshold)
        min_samples: Minimum cluster size
    
    Returns:
        labels: Cluster assignment (-1 for outliers)
    """
    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(embeddings)
    distance_matrix = 1 - sim_matrix
    
    # DBSCAN
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='precomputed'
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    return labels, sim_matrix


def cluster_connected_components(embeddings, threshold=0.65):
    """
    Graph-based clustering: Connect faces if similarity > threshold.
    Simple and interpretable approach.
    
    Args:
        embeddings: (N, 512) normalized face embeddings
        threshold: Similarity threshold for connecting faces
    
    Returns:
        labels: Cluster assignment
    """
    n = len(embeddings)
    sim_matrix = compute_similarity_matrix(embeddings)
    
    # Create adjacency matrix
    adjacency = (sim_matrix >= threshold).astype(int)
    np.fill_diagonal(adjacency, 0)
    
    # Find connected components using Union-Find
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Connect all pairs above threshold
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                union(i, j)
    
    # Assign cluster labels
    labels = np.array([find(i) for i in range(n)])
    
    # Relabel to consecutive integers
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    
    return labels, sim_matrix


def analyze_clusters(labels, sim_matrix):
    """
    Analyze clustering quality.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = np.sum(labels == -1)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of outliers: {n_outliers}")
    
    # Cluster size distribution
    cluster_sizes = Counter(labels[labels != -1])
    print(f"\nCluster size distribution:")
    for size, count in sorted(Counter(cluster_sizes.values()).items()):
        print(f"  {count} clusters with {size} face(s)")
    
    # Intra-cluster similarity (should be high)
    intra_sims = []
    for label in set(labels):
        if label == -1:
            continue
        mask = labels == label
        if mask.sum() > 1:
            cluster_sims = sim_matrix[mask][:, mask]
            # Get upper triangle (exclude diagonal)
            triu_indices = np.triu_indices_from(cluster_sims, k=1)
            intra_sims.extend(cluster_sims[triu_indices])
    
    if intra_sims:
        print(f"\nIntra-cluster similarity: {np.mean(intra_sims):.3f} ± {np.std(intra_sims):.3f}")
    
    # Inter-cluster similarity (should be low)
    inter_sims = []
    unique_labels = [l for l in set(labels) if l != -1]
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            mask1 = labels == label1
            mask2 = labels == label2
            inter_sims.extend(sim_matrix[mask1][:, mask2].flatten())
    
    if inter_sims:
        print(f"Inter-cluster similarity: {np.mean(inter_sims):.3f} ± {np.std(inter_sims):.3f}")
    
    return cluster_sizes


def find_optimal_threshold(embeddings, thresholds=np.arange(0.5, 0.8, 0.02)):
    """
    Try different thresholds and analyze results.
    """
    results = []
    
    for threshold in thresholds:
        labels, _ = cluster_connected_components(embeddings, threshold)
        n_clusters = len(set(labels))
        cluster_sizes = Counter(labels)
        avg_size = np.mean(list(cluster_sizes.values()))
        
        results.append({
            'threshold': threshold,
            'n_clusters': n_clusters,
            'avg_size': avg_size,
            'max_size': max(cluster_sizes.values()),
            'singletons': sum(1 for v in cluster_sizes.values() if v == 1)
        })
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(thresholds, [r['n_clusters'] for r in results])
    axes[0, 0].set_xlabel('Similarity Threshold')
    axes[0, 0].set_ylabel('Number of Clusters')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(thresholds, [r['avg_size'] for r in results])
    axes[0, 1].set_xlabel('Similarity Threshold')
    axes[0, 1].set_ylabel('Average Cluster Size')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(thresholds, [r['max_size'] for r in results])
    axes[1, 0].set_xlabel('Similarity Threshold')
    axes[1, 0].set_ylabel('Largest Cluster Size')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(thresholds, [r['singletons'] for r in results])
    axes[1, 1].set_xlabel('Similarity Threshold')
    axes[1, 1].set_ylabel('Number of Singleton Clusters')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved threshold_analysis.png")
    
    return results


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # Load your embeddings
    # embeddings = np.load('lvface_embeddings.npy')  # Shape: (780, 512)
    
    # Example with random data (replace with your actual embeddings)
    np.random.seed(42)
    n_faces = 780
    n_dims = 512
    embeddings = np.random.randn(n_faces, n_dims)
    # Normalize to unit sphere (LVFace already does this)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    
    print("=" * 60)
    print("METHOD 1: Hierarchical Clustering (Recommended)")
    print("=" * 60)
    labels_hier, sim_matrix = cluster_hierarchical(
        embeddings, 
        threshold=0.65,  # Same person if similarity > 0.65
        linkage='average'
    )
    analyze_clusters(labels_hier, sim_matrix)
    
    
    print("\n" + "=" * 60)
    print("METHOD 2: Connected Components (Simple & Fast)")
    print("=" * 60)
    labels_cc, sim_matrix = cluster_connected_components(
        embeddings,
        threshold=0.65
    )
    analyze_clusters(labels_cc, sim_matrix)
    
    
    print("\n" + "=" * 60)
    print("METHOD 3: DBSCAN (Detects Outliers)")
    print("=" * 60)
    labels_dbscan, sim_matrix = cluster_dbscan(
        embeddings,
        eps=0.35,  # 1 - 0.65 = 0.35 distance
        min_samples=2
    )
    analyze_clusters(labels_dbscan, sim_matrix)
    
    
    print("\n" + "=" * 60)
    print("Finding Optimal Threshold...")
    print("=" * 60)
    results = find_optimal_threshold(embeddings)
