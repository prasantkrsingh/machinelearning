# ===============================
# Pure Python Implementation
# ===============================

# -------------------------------
# Question 1: K-Means Clustering
# -------------------------------

print("=== Question 1: K-Means Clustering ===\n")

# Part A: Small 2D dataset
print("Part A: Small Dataset\n")
X = [[1,2], [1,4], [1,0], [10,2], [10,4], [10,0]]
centroids = [[1,2],[10,2]]
k = 2

def euclidean(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

def mean_point(points):
    n = len(points)
    if n==0:
        return [0,0]
    return [sum(p[0] for p in points)/n, sum(p[1] for p in points)/n]

for iteration in range(10):
    clusters = [[] for _ in range(k)]
    assignments = []
    for point in X:
        distances = [euclidean(point, c) for c in centroids]
        cluster_idx = distances.index(min(distances))
        clusters[cluster_idx].append(point)
        assignments.append(cluster_idx)
    new_centroids = [mean_point(cluster) for cluster in clusters]
    if new_centroids == centroids:
        break
    centroids = new_centroids

print("Cluster assignments:", assignments)
print("Final centroids:", centroids)

# -------------------------------
# Question 2: K-Medoids Clustering
# -------------------------------

print("\n=== Question 2: K-Medoids Clustering ===\n")

# Part A: Small 1D dataset
print("Part A: Small 1D Dataset\n")
X1D = [1,2,2,3,10,11,12]
medoids = [2,11]
k = 2

def manhattan(a,b):
    return abs(a-b)

for iteration in range(10):
    clusters = [[] for _ in range(k)]
    assignments = []
    for x in X1D:
        distances = [manhattan(x,m) for m in medoids]
        idx = distances.index(min(distances))
        clusters[idx].append(x)
        assignments.append(idx)
    new_medoids = []
    for cluster in clusters:
        min_cost = float('inf')
        medoid = cluster[0]
        for candidate in cluster:
            cost = sum(manhattan(candidate, p) for p in cluster)
            if cost < min_cost:
                min_cost = cost
                medoid = candidate
        new_medoids.append(medoid)
    if new_medoids == medoids:
        break
    medoids = new_medoids

total_cost = sum(manhattan(X1D[i], medoids[assignments[i]]) for i in range(len(X1D)))
print("Final medoids:", medoids)
print("Cluster assignments:", assignments)
print("Total cost:", total_cost)


# -------------------------------
# Question 3: Multiple Regression (OLS)
# -------------------------------

print("\n=== Question 3: Multiple Regression (OLS) ===\n")

# Part A: Student scores
print("Part A: Student Scores\n")
data = [
    [1,8,35.2],
    [2,7,37.9],
    [3,7,44.3],
    [4,6,46.8],
    [5,6,53.1],
    [6,5,55.9],
    [7,5,62.2],
    [8,4,64.0]
]

# X = [[Hours_Study, Hours_Sleep]] , y = Score
X = [[row[0], row[1], 1] for row in data]  # add 1 for intercept
y = [row[2] for row in data]

# Ordinary Least Squares: beta = (X^T X)^-1 X^T y
def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

def mat_mult(A,B):
    result = [[0]*len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k]*B[k][j]
    return result

def mat_inv_3x3(M):
    # inverse of 3x3 matrix
    a,b,c = M[0]
    d,e,f = M[1]
    g,h,i = M[2]
    det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
    if det == 0:
        return None
    invdet = 1/det
    inv = [
        [(e*i - f*h)*invdet, (c*h - b*i)*invdet, (b*f - c*e)*invdet],
        [(f*g - d*i)*invdet, (a*i - c*g)*invdet, (c*d - a*f)*invdet],
        [(d*h - e*g)*invdet, (b*g - a*h)*invdet, (a*e - b*d)*invdet]
    ]
    return inv

# Compute beta = (X^T X)^-1 X^T y
Xt = transpose(X)
XtX = mat_mult(Xt, X)
XtX_inv = mat_inv_3x3(XtX)
Xty = mat_mult(Xt, [[val] for val in y])
beta_matrix = mat_mult(XtX_inv, Xty)
beta = [b[0] for b in beta_matrix]
print("Coefficients (β1, β2, β0):", beta)

# R^2 calculation
y_mean = sum(y)/len(y)
ss_tot = sum((yi - y_mean)**2 for yi in y)
y_pred = [beta[0]*xi[0] + beta[1]*xi[1] + beta[2] for xi in X]
ss_res = sum((y[i] - y_pred[i])**2 for i in range(len(y)))
r2 = 1 - ss_res/ss_tot
print("R^2:", r2)
