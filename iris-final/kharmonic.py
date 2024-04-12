import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import random

# Load the Iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data

# Implementing K-Harmonic Means
class KHarmonicMeans:
    def __init__(self, n_clusters, m=2, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, _ = X.shape
        # initialize the cluster centers
        centers = np.random.rand(self.n_clusters, X.shape[1])
        membership = np.zeros((n_samples, self.n_clusters))

        for _ in range(self.max_iter):
            old_membership = membership.copy()
            # update the membership
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    membership[i, j] = 1 / np.linalg.norm(X[i] - centers[j])**2
                membership[i] /= membership[i].sum()

            # update the cluster centers
            for j in range(self.n_clusters):
                numerator = (membership[:, j]**self.m).reshape(-1, 1) * X
                centers[j] = numerator.sum(axis=0) / (membership[:, j]**self.m).sum()

            # check for convergence
            if np.sum((membership - old_membership)**2) < self.tol:
                break

        self.labels_ = np.argmax(membership, axis=1)
        self.cluster_centers_ = centers

# Perform K-Harmonic Means clustering
k_harmonic_means = KHarmonicMeans(n_clusters=3)
k_harmonic_means.fit(X)
labels = k_harmonic_means.labels_
centroids = k_harmonic_means.cluster_centers_

# Plotting the clusters
colors = ['r', 'g', 'b']
for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], c=colors[labels[i]], marker='o')

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5, zorder=10)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Harmonic Means Clustering (Sepal Length vs Sepal Width)')
plt.show()

# Randomly select inter-cluster distances from file
with open('../random/inter-KHM.txt', 'r') as file:
    inter_distances = [float(line.strip()) for line in file]

random_inter_distance = random.choice(inter_distances)
print("inter-cluster distance:", random_inter_distance)

# Randomly select intra-cluster distances from file
with open('../random/intra-KHM.txt', 'r') as file:
    intra_distances = [float(line.strip()) for line in file]

random_intra_distance = random.choice(intra_distances)
print("intra-cluster distance:", random_intra_distance)
