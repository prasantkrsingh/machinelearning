import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Mall Dataset
data = pd.read_csv("Mall_Customers.csv.xls")

# Select numeric features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Take number of clusters K as input from user
K = int(input("Enter the number of clusters (K): "))

# ----- Traditional K-Means -----
kmeans = KMeans(n_clusters=K, random_state=42)
labels_km = kmeans.fit_predict(X_scaled)
sil_km = silhouette_score(X_scaled, labels_km)
print("Silhouette Score - K-Means:", sil_km)

# ----- Bisecting K-Means -----
def bisecting_kmeans(X, K):
    clusters = [np.arange(len(X))]  # store indices of points per cluster
    final_labels = np.zeros(len(X), dtype=int)

    while len(clusters) < K:
        max_sse = -1
        index_to_split = -1

        # find cluster with max SSE to split
        for i, cluster_indices in enumerate(clusters):
            if len(cluster_indices) > 1:
                km = KMeans(n_clusters=2, random_state=42).fit(X[cluster_indices])
                sse = km.inertia_
                if sse > max_sse:
                    max_sse = sse
                    index_to_split = i
                    best_km = km

        if index_to_split == -1:
            break  # no cluster can be split further

        # split the cluster
        cluster_indices = clusters.pop(index_to_split)
        labels = best_km.labels_
        cluster_0 = cluster_indices[labels == 0]
        cluster_1 = cluster_indices[labels == 1]
        clusters.append(cluster_0)
        clusters.append(cluster_1)

    # Assign labels
    for label, cluster_indices in enumerate(clusters):
        final_labels[cluster_indices] = label

    return final_labels

labels_bkm = bisecting_kmeans(X_scaled, K)
sil_bkm = silhouette_score(X_scaled, labels_bkm)
print("Silhouette Score - Bisecting K-Means:", sil_bkm)

# -------- Visualization --------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels_km, cmap='viridis')
plt.title("K-Means Clustering")

plt.subplot(1,2,2)
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels_bkm, cmap='viridis')
plt.title("Bisecting K-Means Clustering")
plt.show()