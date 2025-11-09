import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import seaborn as sns

# Step 1: Load the dataset (CSV disguised as .xls)
data = pd.read_csv('Mall_Customers.csv.xls')

# Extract numeric features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# ---------------- Question 1: Silhouette Analysis ----------------
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans_pp = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = kmeans_pp.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Plot silhouette score vs k
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs Number of Clusters (K-Means++)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.grid(True)
plt.show()

# Optimal k
optimal_k = k_values[np.argmax(silhouette_scores)]
print(f"Optimal k based on silhouette score: {optimal_k}")

# Silhouette distribution for k = 2, 4, 6
for k in [2, 4, 6]:
    kmeans_pp = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = kmeans_pp.fit_predict(X)
    sample_silhouette_values = silhouette_samples(X, labels)

    plt.figure(figsize=(8, 5))
    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.nipy_spectral(float(i) / k)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        y_lower = y_upper + 10
    plt.title(f'Silhouette Distribution for k = {k}')
    plt.xlabel('Silhouette Coefficient Values')
    plt.ylabel('Cluster')
    plt.axvline(x=np.mean(sample_silhouette_values), color="red", linestyle="--")
    plt.show()

# ---------------- Question 2: Elbow Method ----------------
wcss = []
k_values_elbow = range(1, 13)

for k in k_values_elbow:
    kmeans = KMeans(n_clusters=k, init='random', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values_elbow, wcss, marker='o')
plt.title('Elbow Method: WCSS vs Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# ---------------- Question 3: Initialization Comparison ----------------
k = 5

# Random initialization
kmeans_random = KMeans(n_clusters=k, init='random', random_state=42)
labels_random = kmeans_random.fit_predict(X)
wcss_random = kmeans_random.inertia_
silhouette_random = silhouette_score(X, labels_random)
iterations_random = kmeans_random.n_iter_

# K-Means++ initialization
kmeans_pp = KMeans(n_clusters=k, init='k-means++', random_state=42)
labels_pp = kmeans_pp.fit_predict(X)
wcss_pp = kmeans_pp.inertia_
silhouette_pp = silhouette_score(X, labels_pp)
iterations_pp = kmeans_pp.n_iter_

print("\nComparison for k = 5:")
print(f"Random Init -> WCSS: {wcss_random}, Silhouette: {silhouette_random}, Iterations: {iterations_random}")
print(f"K-Means++ Init -> WCSS: {wcss_pp}, Silhouette: {silhouette_pp}, Iterations: {iterations_pp}")

# Cluster visualizations
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=labels_random, palette='Set1', s=50)
plt.scatter(kmeans_random.cluster_centers_[:, 0], kmeans_random.cluster_centers_[:, 1], s=200, c='yellow', marker='X')
plt.title('Clusters with Random Initialization')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=labels_pp, palette='Set2', s=50)
plt.scatter(kmeans_pp.cluster_centers_[:, 0], kmeans_pp.cluster_centers_[:, 1], s=200, c='yellow', marker='X')
plt.title('Clusters with K-Means++ Initialization')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

plt.tight_layout()
plt.show()