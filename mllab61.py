import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -------------------------
# Load Data
# -------------------------
data = pd.read_csv(r"D:\c\.vscode\ML LAB\Seed_Data.csv") 
X = data.drop(columns=['target'], errors='ignore') 

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Silhouette Analysis + WCSS
# -------------------------
sil_scores = []
wcss = []

print(f"{'k':<5}{'Silhouette Score':<20}{'WCSS':<15}")
for k in range(2, 13):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Silhouette score
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)
    
    # WCSS
    inertia = kmeans.inertia_
    wcss.append(inertia)
    
    print(f"{k:<5}{sil:<20.4f}{inertia:<15.4f}")

# -------------------------
# Silhouette Plot Function
# -------------------------
def plot_silhouette(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_vals = silhouette_samples(X, cluster_labels)
    
    y_lower = 10
    fig, ax = plt.subplots(figsize=(7,5))
    
    for i in range(n_clusters):
        ith_cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
        ith_cluster_silhouette_vals.sort()
        size_cluster_i = ith_cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # space between clusters
    
    ax.set_title(f"Silhouette plot for k={n_clusters}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_score(X, cluster_labels), color="red", linestyle="--")
    ax.set_yticks([])
    plt.show()

# Plot silhouette distributions for k = 2, 4, 8
for k in [2, 4, 8]:
    plot_silhouette(X_scaled, k)

# -------------------------
# Elbow Method
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(range(2,13), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(range(2,13))
plt.grid(True)
plt.show()
