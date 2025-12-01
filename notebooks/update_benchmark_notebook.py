import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell("""
# Clustering Benchmark on UCI Digits Dataset

This notebook benchmarks various clustering algorithms implemented in `datascienceutils` against `sklearn` implementations.

Algorithms:
- **HDBSCAN**: Density-based clustering (from `datascienceutils`)
- **K-Means**: Centroid-based clustering (from `datascienceutils` and `sklearn`)
- **Spectral Clustering**: Graph-based clustering (from `datascienceutils`)
- **K-Medians**: Centroid-based clustering with L1 norm (from `datascienceutils`)
- **DBSCAN**: Density-based clustering (from `sklearn`)

Dataset: UCI Digits (8x8 images of digits)
"""))

nb.cells.append(nbf.v4.new_code_cell("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans as SklearnKMeans, DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
import datascienceutils as dsu
import time

# Load data
digits = load_digits()
data = digits.data
target = digits.target

print(f"Data shape: {data.shape}")
print(f"Classes: {np.unique(target)}")

# Reduce dimensionality for visualization
tsne = TSNE(n_components=2, random_state=42)
data_2d = tsne.fit_transform(data)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=target, cmap='tab10', s=10, alpha=0.6)
plt.colorbar(scatter)
plt.title("Ground Truth (t-SNE)")
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
## Benchmarking Function
"""))

nb.cells.append(nbf.v4.new_code_cell("""
def benchmark_algorithm(name, cluster_func, data, true_labels):
    start_time = time.time()
    try:
        labels = cluster_func(data)
        elapsed = time.time() - start_time
        
        # Handle noise (-1) for silhouette score
        if len(np.unique(labels)) > 1:
            # Filter out noise for silhouette calculation if present
            mask = labels != -1
            if np.sum(mask) > 1:
                sil_score = silhouette_score(data[mask], labels[mask])
            else:
                sil_score = -1.0
        else:
            sil_score = -1.0
            
        ari = adjusted_rand_score(true_labels, labels)
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        
        print(f"--- {name} ---")
        print(f"Time: {elapsed:.4f}s")
        print(f"Clusters: {n_clusters}")
        print(f"ARI: {ari:.4f}")
        print(f"Silhouette: {sil_score:.4f}")
        
        return {
            'name': name,
            'time': elapsed,
            'ari': ari,
            'silhouette': sil_score,
            'labels': labels
        }
    except Exception as e:
        print(f"--- {name} ---")
        print(f"Failed: {e}")
        return None
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
## Run Benchmarks
"""))

nb.cells.append(nbf.v4.new_code_cell("""
results = []

# 1. HDBSCAN (datascienceutils)
def run_hdbscan(X):
    res = dsu.hdbscan_cluster(X, min_cluster_size=15)
    return np.array(res.labels)

results.append(benchmark_algorithm("DSU HDBSCAN", run_hdbscan, data, target))

# 2. K-Means (datascienceutils)
def run_dsu_kmeans(X):
    res = dsu.kmeans_cluster(X, n_clusters=10, seed=42)
    return np.array(res.labels)

results.append(benchmark_algorithm("DSU K-Means", run_dsu_kmeans, data, target))

# 3. K-Medians (datascienceutils)
def run_dsu_kmedians(X):
    res = dsu.kmedians_cluster(X, n_clusters=10, seed=42)
    return np.array(res.labels)

results.append(benchmark_algorithm("DSU K-Medians", run_dsu_kmedians, data, target))

# 4. Spectral Clustering (datascienceutils)
# Note: Spectral clustering can be slow on large datasets
def run_dsu_spectral(X):
    # Use a subset or just run it (1797 samples is fine)
    res = dsu.spectral_cluster(X, n_clusters=10, gamma=0.001, seed=42)
    return np.array(res.labels)

results.append(benchmark_algorithm("DSU Spectral", run_dsu_spectral, data, target))

# 5. Sklearn K-Means
def run_sklearn_kmeans(X):
    kmeans = SklearnKMeans(n_clusters=10, random_state=42, n_init=10)
    return kmeans.fit_predict(X)

results.append(benchmark_algorithm("Sklearn K-Means", run_sklearn_kmeans, data, target))

# 6. Sklearn DBSCAN
def run_sklearn_dbscan(X):
    # DBSCAN is sensitive to eps, need to tune
    dbscan = SklearnDBSCAN(eps=20, min_samples=5)
    return dbscan.fit_predict(X)

results.append(benchmark_algorithm("Sklearn DBSCAN", run_sklearn_dbscan, data, target))
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
## Visualization of Results
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# Filter failed runs
valid_results = [r for r in results if r is not None]

# Plot clusters
n_algs = len(valid_results)
rows = (n_algs + 1) // 2
cols = 2

plt.figure(figsize=(15, 5 * rows))

for i, res in enumerate(valid_results):
    plt.subplot(rows, cols, i+1)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=res['labels'], cmap='tab10', s=10, alpha=0.6)
    plt.title(f"{res['name']} (ARI: {res['ari']:.2f})")
    
plt.tight_layout()
plt.show()

# Compare Metrics
names = [r['name'] for r in valid_results]
aris = [r['ari'] for r in valid_results]
times = [r['time'] for r in valid_results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.bar(names, aris, color='skyblue')
ax1.set_title("Adjusted Rand Index (Higher is Better)")
ax1.tick_params(axis='x', rotation=45)

ax2.bar(names, times, color='salmon')
ax2.set_title("Execution Time (Lower is Better)")
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
"""))

with open('/home/roci/playspace/data-science-utils/notebooks/09_clustering_benchmark_uci.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Benchmark notebook updated.")
