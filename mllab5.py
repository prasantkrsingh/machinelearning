import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# -------------------- Utility Functions --------------------
def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def assign_clusters(points, centroids):
    clusters = [[] for _ in centroids]
    for p in points:
        distances = [euclidean(p, c) for c in centroids]
        min_index = distances.index(min(distances))
        clusters[min_index].append(p)
    return clusters

def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if cluster:
            x_mean = sum(p[0] for p in cluster) / len(cluster)
            y_mean = sum(p[1] for p in cluster) / len(cluster)
            new_centroids.append((x_mean, y_mean))
    return new_centroids

def kmeans(points, k, initial_centroids, max_iter=100):
    centroids = initial_centroids
    for _ in range(max_iter):
        clusters = assign_clusters(points, centroids)
        new_centroids = update_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return clusters, centroids

def print_clusters(clusters, centroids):
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {len(cluster)} points")
        print(f" Centroid: {centroids[i]}")
    print("-"*40)

def plot_clusters(clusters, centroids, title, xlabel="X", ylabel="Y"):
    colors = ["blue", "green", "orange", "purple", "brown"]
    for i, cluster in enumerate(clusters):
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        plt.scatter(xs, ys, color=colors[i % len(colors)], label=f"Cluster {i+1}")
    # plot centroids
    cx = [c[0] for c in centroids]
    cy = [c[1] for c in centroids]
    plt.scatter(cx, cy, color="red", marker="X", s=200, label="Centroids")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# -------------------- Part A --------------------
print("===== Part A: Given Dataset =====")
points_A = [(1,2), (1,4), (1,0), (10,2), (10,4), (10,0)]
initial_centroids_A = [(1,2), (10,2)]
clusters_A, centroids_A = kmeans(points_A, 2, initial_centroids_A)

print_clusters(clusters_A, centroids_A)
plot_clusters(clusters_A, centroids_A, "Part A: K-Means Clustering (k=2)")

# -------------------- Part B --------------------
print("\n===== Part B: Iris Dataset (sepal length vs petal length) =====")

iris = load_iris()
# Select sepal length (col 0) and petal length (col 2)
iris_points = [(float(x[0]), float(x[2])) for x in iris.data]

k = 3
initial_centroids_B = iris_points[:k]
clusters_B, centroids_B = kmeans(iris_points, k, initial_centroids_B)

print_clusters(clusters_B, centroids_B)

plot_clusters(
    clusters_B, 
    centroids_B, 
    "Part B: Iris K-Means (k=3)", 
    xlabel="Sepal length", 
    ylabel="Petal length"
)
