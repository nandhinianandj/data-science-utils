import nbformat as nbf
import os
import pandas as pd
import numpy as np

# Create dummy data if not exists
data_path = 'data/Customer_Segmentation.csv'
if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.exists(data_path):
    print(f"Creating synthetic dataset at {data_path}")
    np.random.seed(42)
    n_samples = 2000
    
    data = {
        'ID': range(1, n_samples + 1),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Ever_Married': np.random.choice(['Yes', 'No'], n_samples),
        'Age': np.random.randint(18, 89, n_samples),
        'Graduated': np.random.choice(['Yes', 'No'], n_samples),
        'Profession': np.random.choice(['Artist', 'Healthcare', 'Engineer', 'Lawyer', 'Entertainment', 'Executive', 'Doctor', 'Homemaker', 'Marketing'], n_samples),
        'Work_Experience': np.random.randint(0, 15, n_samples),
        'Spending_Score': np.random.choice(['Low', 'Average', 'High'], n_samples),
        'Family_Size': np.random.randint(1, 10, n_samples),
        'Var_1': np.random.choice(['Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7'], n_samples),
        'Segmentation': np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell("""
# Customer Segmentation Clustering Benchmark

This notebook benchmarks various clustering algorithms on the Customer Segmentation dataset.

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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN
import datascienceutils as dsu
import time

# Load data
df = pd.read_csv('../data/Customer_Segmentation.csv')
print(f"Data shape: {df.shape}")
print(df.head())
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
## Preprocessing
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# Drop ID
df = df.drop('ID', axis=1)

# Handle missing values (simple imputation)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Encode categorical variables
# For clustering, One-Hot Encoding is often better, but for high cardinality, Label Encoding or Target Encoding might be used.
# Here we use Label Encoding for simplicity and to keep dimensionality low for distance metrics.
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    if col != 'Segmentation': # Keep target for reference if needed, though we don't use it for clustering
        df[col] = le.fit_transform(df[col])

# Scale features
scaler = StandardScaler()
features = [c for c in df.columns if c != 'Segmentation']
X = scaler.fit_transform(df[features])

print("Scaled data shape:", X.shape)
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
## Visualization (PCA & t-SNE)
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

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
# Spectral can be slow, use subset if needed, but 2000 samples is fine
res_spectral = dsu.spectral_cluster(X, n_clusters=4, gamma=0.1, seed=42)
results.append(evaluate_clusters("Spectral (DSU)", np.array(res_spectral.labels), X, time.time() - start))

# 5. DBSCAN (Sklearn)
start = time.time()
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)
results.append(evaluate_clusters("DBSCAN (Sklearn)", labels_dbscan, X, time.time() - start))
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""
## Results Visualization
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# Plot clusters on t-SNE
n_algs = len(results)
rows = (n_algs + 1) // 2
cols = 2

plt.figure(figsize=(15, 5 * rows))

for i, res in enumerate(results):
    plt.subplot(rows, cols, i+1)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=res['labels'], cmap='tab10', s=10, alpha=0.6)
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

with open('/home/roci/playspace/data-science-utils/notebooks/10_customer_segmentation_benchmark.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created.")
