import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.metrics.pairwise import pairwise_distances

# Load the wine dataset
wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)

# Select two features
feature1 = 'alcohol'
feature2 = 'flavanoids'

X = data[[feature1, feature2]]

# K Harmonic means clustering
def k_harmonic_means(X, k, max_iter=100):
    centers = X[np.random.choice(range(len(X)), k, replace=False)]
    for _ in range(max_iter):
        distances = pairwise_distances(X, centers)
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return labels, centers

k_harmonic_labels, k_harmonic_centers = k_harmonic_means(X.values, 3)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[feature1], X[feature2], c=k_harmonic_labels, cmap='viridis')
plt.scatter(k_harmonic_centers[:, 0], k_harmonic_centers[:, 1], marker='o', c='red', s=200, label='Centroids')
plt.title('K Harmonic Means Clustering')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()
plt.xlim(X[feature1].min() - 1, X[feature1].max() + 1)
plt.ylim(X[feature2].min() - 1, X[feature2].max() + 1)
plt.show()

# Calculate inter-cluster and intra-cluster distances for 3 clusters
k_harmonic_distances = pairwise_distances(X, k_harmonic_centers)

intra_cluster_k_harmonic = np.min(k_harmonic_distances, axis=1)
inter_cluster_k_harmonic = np.mean(np.min(k_harmonic_distances, axis=0))

print("Inter-cluster distance for K Harmonic Means:", inter_cluster_k_harmonic)
print("Intra-cluster distance for K Harmonic Means:", np.mean(intra_cluster_k_harmonic))
