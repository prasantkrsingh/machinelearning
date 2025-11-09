# kmedoids_manhattan.py
import math
import random
import matplotlib.pyplot as plt

# -------------------- Distance --------------------
def manhattan(p1, p2):
    """Manhattan distance supporting scalars (1D) and iterables (tuples/lists)."""
    if isinstance(p1, (int, float)) and isinstance(p2, (int, float)):
        return abs(p1 - p2)
    return sum(abs(a - b) for a, b in zip(p1, p2))


# -------------------- Core K-Medoids Functions --------------------
def assign_clusters(points, medoids):
    k = len(medoids)
    clusters = {i: [] for i in range(k)}
    total_cost = 0.0
    for p in points:
        distances = [manhattan(p, m) for m in medoids]
        min_idx = distances.index(min(distances))
        clusters[min_idx].append(p)
        total_cost += float(distances[min_idx])
    return clusters, total_cost


def compute_new_medoids(clusters, medoids):
    new_medoids = []
    for i in range(len(medoids)):
        cluster = clusters.get(i, [])
        if not cluster:
            new_medoids.append(medoids[i])  # keep old medoid
            continue
        best_point, best_cost = cluster[0], float("inf")
        for candidate in cluster:
            cost = sum(manhattan(candidate, other) for other in cluster)
            if cost < best_cost:
                best_cost, best_point = cost, candidate
        new_medoids.append(best_point)
    return new_medoids


def kmedoids(points, k, initial_medoids, max_iter=100, verbose=False):
    medoids = list(initial_medoids)
    for it in range(1, max_iter + 1):
        clusters, cost = assign_clusters(points, medoids)
        if verbose:
            print(f"Iteration {it}:")
            print("  Medoids:", medoids)
            for idx in range(len(medoids)):
                print(f"   Cluster {idx}: {clusters.get(idx, [])}")
            print(f"  Total cost: {cost:.4f}")
            print("-" * 40)
        new_medoids = compute_new_medoids(clusters, medoids)
        if new_medoids == medoids:
            break
        medoids = new_medoids
    clusters, cost = assign_clusters(points, medoids)
    return clusters, medoids, cost


# -------------------- Printing Helpers --------------------
def print_result_partA(clusters, medoids, cost):
    print("\n Part A Result")
    print("Final medoids:", medoids)
    print(f"Total cost: {cost:.4f}")
    for i, cl in clusters.items():
        print(f" Cluster {i+1}: {cl}  (size={len(cl)})")
    print("=" * 30 + "\n")


def print_result_partB(clusters, medoids, cost):
    print("\nPart B Result ")
    print("Final medoids:", medoids)
    print(f"Total cost: {cost:.4f}")
    for i, cl in clusters.items():
        print(f" Cluster {i+1} size: {len(cl)}")
    print("=" * 30 + "\n")


# -------------------- Plot Helpers --------------------
def plot_clusters_1d(clusters, medoids, title="1D K-Medoids"):
    plt.figure(figsize=(8, 3))
    for i, cluster in clusters.items():
        xs = [p for p in cluster]
        ys = [0] * len(xs)
        plt.scatter(xs, ys, label=f"Cluster {i+1}", s=50)
    mx = [m for m in medoids]
    my = [0] * len(mx)
    plt.scatter(mx, my, marker='X', s=150, label='Medoids', edgecolors='black')
    plt.title(title)
    plt.yticks([])
    plt.xlabel("Value")
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_clusters_2d(clusters, medoids, title="2D K-Medoids", xlabel="X", ylabel="Y"):
    plt.figure(figsize=(7, 5))
    for i, cluster in clusters.items():
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        plt.scatter(xs, ys, label=f"Cluster {i+1}", s=40)
    mx = [m[0] for m in medoids]
    my = [m[1] for m in medoids]
    plt.scatter(mx, my, marker='X', s=150, label='Medoids', edgecolors='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# -------------------- Main --------------------
def main():
    # ------- Part A -------
    print("PART A: 1D dataset [1,2,2,3,10,11,12], k=2, initial medoids {2,11}")
    points_A = [1, 2, 2, 3, 10, 11, 12]
    initial_medoids_A = [2, 11]
    clusters_A, medoids_A, cost_A = kmedoids(points_A, k=2,
                                             initial_medoids=initial_medoids_A,
                                             max_iter=50, verbose=True)
    print_result_partA(clusters_A, medoids_A, cost_A)
    plot_clusters_1d(clusters_A, medoids_A, title="Part A: K-Medoids (1D)")

    # ------- Part B -------
    print("PART B: Iris (sepal length vs petal length), k=3, Manhattan distance")
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        iris_points = [(float(x[0]), float(x[2])) for x in iris.data]
        print("Loaded full Iris dataset (150 samples).")
    except Exception:
        iris_points = [(5.1, 1.4), (4.9, 1.4), (4.7, 1.3),
                       (7.0, 4.7), (6.4, 4.5), (6.9, 4.9),
                       (5.5, 1.5), (5.7, 1.7), (6.3, 4.9)]
        print("sklearn not available — using small sample.")

    k = 3
    random.seed(0)
    initial_medoids_B = random.sample(iris_points, k)
    clusters_B, medoids_B, cost_B = kmedoids(iris_points, k=k,
                                             initial_medoids=initial_medoids_B,
                                             max_iter=200, verbose=False)
    print_result_partB(clusters_B, medoids_B, cost_B)
    plot_clusters_2d(clusters_B, medoids_B, title="Part B: K-Medoids (Iris)",
                     xlabel="Sepal length", ylabel="Petal length")


if __name__ == "__main__":
    main()
