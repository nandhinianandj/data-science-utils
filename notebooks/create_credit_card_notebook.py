import nbformat as nbf
import os
import pandas as pd
import numpy as np

# Create dummy data if not exists
data_path = 'data/CC GENERAL.csv'
if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.exists(data_path):
    print(f"Creating synthetic dataset at {data_path}")
    np.random.seed(42)
    n_samples = 2000
    
    data = {
        'CUST_ID': [f'C{i}' for i in range(10000, 10000 + n_samples)],
        'BALANCE': np.random.exponential(1500, n_samples),
        'BALANCE_FREQUENCY': np.random.uniform(0, 1, n_samples),
        'PURCHASES': np.random.exponential(1000, n_samples),
        'ONEOFF_PURCHASES': np.random.exponential(600, n_samples),
        'INSTALLMENTS_PURCHASES': np.random.exponential(400, n_samples),
        'CASH_ADVANCE': np.random.exponential(1000, n_samples),
        'PURCHASES_FREQUENCY': np.random.uniform(0, 1, n_samples),
        'ONEOFF_PURCHASES_FREQUENCY': np.random.uniform(0, 1, n_samples),
        'PURCHASES_INSTALLMENTS_FREQUENCY': np.random.uniform(0, 1, n_samples),
        'CASH_ADVANCE_FREQUENCY': np.random.uniform(0, 1, n_samples),
        'CASH_ADVANCE_TRX': np.random.randint(0, 15, n_samples),
        'PURCHASES_TRX': np.random.randint(0, 20, n_samples),
        'CREDIT_LIMIT': np.random.choice([1000, 2500, 5000, 7500, 10000, 15000], n_samples),
        'PAYMENTS': np.random.exponential(1700, n_samples),
        'MINIMUM_PAYMENTS': np.random.exponential(800, n_samples), # Some will be NaN in real data, but here we generate values
        'PRC_FULL_PAYMENT': np.random.uniform(0, 1, n_samples),
        'TENURE': np.random.choice([6, 12], n_samples)
    }
    
    # Introduce some NaNs in MINIMUM_PAYMENTS to mimic real data
    min_pay = data['MINIMUM_PAYMENTS']
    min_pay[np.random.choice(n_samples, 50, replace=False)] = np.nan
    data['MINIMUM_PAYMENTS'] = min_pay
    
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell("""
# Credit Card Clustering Benchmark

This notebook benchmarks various clustering algorithms on the Credit Card dataset.

Algorithms:
- **HDBSCAN** (dsu)
- **K-Means** (dsu)
- **K-Medians** (dsu)
- **Spectral Clustering** (dsu)
- **DBSCAN** (sklearn)
"""))

nb.cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN
import datascienceutils as dsu
import time

# Load data
df = pd.read_csv('../data/CC GENERAL.csv')
print(f"Data shape: {df.shape}")
print(df.head())
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
## Preprocessing
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# Drop CUST_ID
if 'CUST_ID' in df.columns:
    df = df.drop('CUST_ID', axis=1)

# Handle missing values
df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(df)

print("Scaled data shape:", X.shape)
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
## Visualization (PCA & t-SNE)
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE (use subset if data is too large, e.g., > 5000 samples)
if X.shape[0] > 5000:
    indices = np.random.choice(X.shape[0], 5000, replace=False)
    X_tsne_input = X[indices]
else:
    X_tsne_input = X
    
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_tsne_input)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10, alpha=0.5)
plt.title("PCA Projection")

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=10, alpha=0.5)
plt.title("t-SNE Projection")
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
## Benchmarking
"""))

nb.cells.append(nbf.v4.new_code_cell("""
results = []

def evaluate_clusters(name, labels, data, time_taken):
    # Filter noise for metrics
    if len(np.unique(labels)) > 1:
        mask = labels != -1
        if np.sum(mask) > 1:
            sil = silhouette_score(data[mask], labels[mask])
            db = davies_bouldin_score(data[mask], labels[mask])
        else:
            sil = -1
            db = 10
    else:
        sil = -1
        db = 10
        
    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    
    print(f"--- {name} ---")
    print(f"Time: {time_taken:.4f}s")
    print(f"Clusters: {n_clusters}")
    print(f"Silhouette: {sil:.4f}")
    print(f"Davies-Bouldin: {db:.4f}")
    
    return {
        'name': name,
        'time': time_taken,
        'clusters': n_clusters,
        'silhouette': sil,
        'davies_bouldin': db,
        'labels': labels
    }

# 1. HDBSCAN
start = time.time()
res_hdbscan = dsu.hdbscan_cluster(X, min_cluster_size=30)
results.append(evaluate_clusters("HDBSCAN", np.array(res_hdbscan.labels), X, time.time() - start))

# 2. K-Means (DSU)
start = time.time()
res_kmeans = dsu.kmeans_cluster(X, n_clusters=4, seed=42)
results.append(evaluate_clusters("K-Means (DSU)", np.array(res_kmeans.labels), X, time.time() - start))

# 3. K-Medians (DSU)
start = time.time()
res_kmedians = dsu.kmedians_cluster(X, n_clusters=4, seed=42)
results.append(evaluate_clusters("K-Medians (DSU)", np.array(res_kmedians.labels), X, time.time() - start))

# 4. Spectral Clustering (DSU)
start = time.time()
# Spectral can be slow, use subset if needed
if X.shape[0] > 2000:
    indices = np.random.choice(X.shape[0], 2000, replace=False)
    X_spectral = X[indices]
else:
    X_spectral = X
    
res_spectral = dsu.spectral_cluster(X_spectral, n_clusters=4, gamma=0.1, seed=42)
# Map labels back to full dataset (placeholder -1 for others) or just evaluate on subset
# For simplicity, we'll just evaluate on the subset for spectral
print("--- Spectral (DSU) [Subset] ---")
evaluate_clusters("Spectral (DSU)", np.array(res_spectral.labels), X_spectral, time.time() - start)
# Note: Not adding to results list for full comparison plots due to size mismatch, or could handle it.
# Let's skip adding to main results for simplicity of plotting, or add with a note.

# 5. DBSCAN (Sklearn)
start = time.time()
dbscan = DBSCAN(eps=2.0, min_samples=5) # eps needs tuning for scaled data
labels_dbscan = dbscan.fit_predict(X)
results.append(evaluate_clusters("DBSCAN (Sklearn)", labels_dbscan, X, time.time() - start))
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
## Results Visualization
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# Plot clusters on t-SNE (using subset if needed)
# We use X_tsne which might be a subset
if X.shape[0] > 5000:
    # We need labels for the subset used for t-SNE
    # This is tricky if we ran clustering on full data.
    # For visualization, let's just use the indices we selected for t-SNE
    plot_indices = indices
    plot_labels = [r['labels'][plot_indices] for r in results]
else:
    plot_labels = [r['labels'] for r in results]

n_algs = len(results)
rows = (n_algs + 1) // 2
cols = 2

plt.figure(figsize=(15, 5 * rows))

for i, res in enumerate(results):
    plt.subplot(rows, cols, i+1)
    labels = res['labels']
    if len(labels) > len(X_tsne):
        # If we have full labels but t-SNE is subset
        labels = labels[indices]
        
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=10, alpha=0.6)
    plt.title(f"{res['name']} (Sil: {res['silhouette']:.2f})")
    
plt.tight_layout()
plt.show()

# Comparison Bar Charts
names = [r['name'] for r in results]
sils = [r['silhouette'] for r in results]
dbs = [r['davies_bouldin'] for r in results]
times = [r['time'] for r in results]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.bar(names, sils, color='skyblue')
ax1.set_title("Silhouette Score (Higher is Better)")
ax1.tick_params(axis='x', rotation=45)

ax2.bar(names, dbs, color='lightgreen')
ax2.set_title("Davies-Bouldin Index (Lower is Better)")
ax2.tick_params(axis='x', rotation=45)

ax3.bar(names, times, color='salmon')
ax3.set_title("Execution Time (Lower is Better)")
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
"""))

with open('/home/roci/playspace/data-science-utils/notebooks/11_credit_card_clustering.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created.")
