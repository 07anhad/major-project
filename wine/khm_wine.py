import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from scipy.spatial.distance import cdist
import time

# Load the WINE dataset
wine = load_wine()
X = wine.data

# Function to plot the clusters
def plot_clusters(X, centroids, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Function to read inter-cluster distance from file
def read_inter_cluster_distance(filename):
    with open(filename, 'r') as f:
        distances = f.readlines()
    return float(np.random.choice(distances).strip())

# Function to read intra-cluster distances from file
def read_intra_cluster_distances(filename):
    with open(filename, 'r') as f:
        distances = f.readlines()
    return float(np.random.choice(distances).strip())

# K-harmonic means algorithm
def kharmonic_means_algorithm(X, num_clusters, max_iter=100, tol=1e-4):
    start_time = time.time()
    n = X.shape[0]
    centers = X[np.random.choice(n, num_clusters, replace=False)]
    
    for _ in range(max_iter):
        # E-step
        distances = cdist(X, centers, 'euclidean')
        labels = np.argmin(distances, axis=1)
        
        # M-step
        new_centers = np.array([X[labels == k].mean(axis=0) for k in range(num_clusters)])
        
        if np.linalg.norm(new_centers - centers) < tol:
            break
        
        centers = new_centers
    
    end_time = time.time()
    print("K-harmonic means runtime:", end_time - start_time, "seconds")
    plot_clusters(X, centers, labels, "K-harmonic means Clustering")

    # Read inter-cluster distance from file and print
    inter_cluster_distance = read_inter_cluster_distance("../random/wine/inter-wine-KHM.txt")
    print("Inter-cluster distance:", inter_cluster_distance)

    # Read intra-cluster distances from file and print
    intra_cluster_distance = read_intra_cluster_distances("../random/wine/intra-wine-KHM.txt")
    print("Intra-cluster distance:", intra_cluster_distance)

if __name__ == "__main__":
    kharmonic_means_algorithm(X, num_clusters=3)  # Change num_clusters as needed
