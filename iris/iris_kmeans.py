import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import random
import time

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the KMeans class
class KMeansCustom:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = np.array(random.sample(list(X), self.n_clusters))
        self.labels = np.zeros(X.shape[0])

        start_time = time.time()

        for _ in range(self.max_iter):
            # Assign each data point to the nearest centroid
            distances = np.linalg.norm(X[:, None, :] - self.centroids, axis=2)
            new_labels = np.argmin(distances, axis=1)

            # Check convergence
            if np.all(self.labels == new_labels):
                break

            # Update centroids
            for i in range(self.n_clusters):
                cluster_points = X[new_labels == i]
                self.centroids[i] = np.mean(cluster_points, axis=0)

            self.labels = new_labels

        end_time = time.time()
        self.running_time = end_time - start_time
        print(f"Running time: {self.running_time:.4f} seconds")

        # Calculate and print intra-cluster and inter-cluster distances
        intra_cluster_distances = []
        inter_cluster_distances = []
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            centroid = self.centroids[i]
            intra_cluster_distance = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
            intra_cluster_distances.append(intra_cluster_distance)
            for j in range(i + 1, self.n_clusters):
                inter_cluster_distance = np.linalg.norm(self.centroids[i] - self.centroids[j])
                inter_cluster_distances.append(inter_cluster_distance)

        print(f"Average Intra-cluster Distance: {np.mean(intra_cluster_distances):.4f}")
        print(f"Average Inter-cluster Distance: {np.mean(inter_cluster_distances):.4f}")

    def predict(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# Perform KMeans clustering
kmeans_custom = KMeansCustom(n_clusters=3)
kmeans_custom.fit(X_scaled)
labels_custom = kmeans_custom.predict(X_scaled)

# Plot the clusters
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i in range(3):
    plt.scatter(X_scaled[labels_custom == i, 0], X_scaled[labels_custom == i, 1], color=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans_custom.centroids[:, 0], kmeans_custom.centroids[:, 1], marker='x', s=200, c='black', label='Centroids')

# Set the ticks and labels for the x-axis
plt.xticks(np.arange(-1.5, 1.6, 0.5))  # Change the tick positions as per your requirement
plt.xlabel('Custom X-axis Label')  # Change the label as per your requirement

plt.ylabel('Sepal Width (scaled)')
plt.title('KMeans Clustering of Iris Dataset (Custom Implementation)')
plt.legend()
plt.show()
