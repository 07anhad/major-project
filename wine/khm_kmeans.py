import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import pairwise_distances

# Load the wine dataset
wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(data)

# Select two features
feature1 = 'alcohol'
feature2 = 'flavanoids'

X = pd.DataFrame(X_imputed, columns=data.columns)[[feature1, feature2]]

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
kmeans_labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot the K-means clusters
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[feature1], X[feature2], c=kmeans_labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], marker='o', c='red', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()

# K Harmonic means clustering
def k_harmonic_means(X, k, max_iter=100):
    centers = X[np.random.choice(range(len(X)), k, replace=False)]
    for _ in range(max_iter):
        distances = pairwise_distances(X, centers)
        harmonic_distances = 1 / (distances + 1e-6)  # Adding a small value to avoid division by zero
        labels = np.argmin(harmonic_distances, axis=1)
        new_centers = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return labels, centers

k_harmonic_labels, k_harmonic_centers = k_harmonic_means(X.values, 3)

# Plot the K Harmonic Means clusters
plt.subplot(1, 2, 2)
plt.scatter(X[feature1], X[feature2], c=k_harmonic_labels, cmap='viridis')
plt.scatter(k_harmonic_centers[:, 0], k_harmonic_centers[:, 1], marker='o', c='red', s=200, label='Centroids')
plt.title('K Harmonic Means Clustering')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()

# Calculate inter-cluster and intra-cluster distances for K-means
kmeans_distances = pairwise_distances(X, centers)
intra_cluster_kmeans = np.min(kmeans_distances, axis=1)
inter_cluster_kmeans = np.mean(np.min(kmeans_distances, axis=0))
print("Inter-cluster distance for K-means:", inter_cluster_kmeans)
print("Intra-cluster distance for K-means:", np.mean(intra_cluster_kmeans))

# Calculate inter-cluster and intra-cluster distances for K Harmonic Means
k_harmonic_distances = pairwise_distances(X, k_harmonic_centers)
intra_cluster_k_harmonic = np.mean(np.min(k_harmonic_distances, axis=1))
inter_cluster_k_harmonic = np.mean(np.min(k_harmonic_distances, axis=0))
print("Inter-cluster distance for K Harmonic Means:", inter_cluster_k_harmonic)
print("Intra-cluster distance for K Harmonic Means:", intra_cluster_k_harmonic)

plt.show()
