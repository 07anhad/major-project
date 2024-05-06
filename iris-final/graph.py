import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np
import random
import time

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data

# Initialize lists to store running times and number of clusters
kmeans_running_times = []
khm_running_times = []
num_clusters = []

# KMeans clustering
for n in range(1, 21):
    start_time = time.time()
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    end_time = time.time()
    running_time = end_time - start_time
    kmeans_running_times.append(running_time)
    num_clusters.append(n)

# Plot KMeans running time
plt.plot(num_clusters, kmeans_running_times, label='K Harmonic Means', color='blue')

# K-Harmonic Means clustering
class KHarmonicMeans:
    def __init__(self, n_clusters, m=2, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, _ = X.shape
        centers = np.random.rand(self.n_clusters, X.shape[1])
        membership = np.zeros((n_samples, self.n_clusters))

        for _ in range(self.max_iter):
            old_membership = membership.copy()
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    membership[i, j] = 1 / np.linalg.norm(X[i] - centers[j])**2
                membership[i] /= membership[i].sum()

            for j in range(self.n_clusters):
                numerator = (membership[:, j]**self.m).reshape(-1, 1) * X
                centers[j] = numerator.sum(axis=0) / (membership[:, j]**self.m).sum()

            if np.sum((membership - old_membership)**2) < self.tol:
                break

        self.labels_ = np.argmax(membership, axis=1)
        self.cluster_centers_ = centers

khm_running_times = []

for n in range(1, 21):
    start_time = time.time()
    k_harmonic_means = KHarmonicMeans(n_clusters=n)
    k_harmonic_means.fit(X)
    end_time = time.time()
    running_time = end_time - start_time
    khm_running_times.append(running_time)

# Plot K-Harmonic Means running time
plt.plot(num_clusters, khm_running_times, label='K Means', color='red')

plt.xlabel('Number of Clusters')
plt.ylabel('Running Time (seconds)')
plt.title('Converging Graph of KMeans and K-Harmonic Means')
plt.legend()
plt.grid(True)
plt.show()
