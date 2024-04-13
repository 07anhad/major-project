import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import time
import random

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

# Function to randomly select one value from inter-cluster distances file
def get_inter_cluster_distance():
    with open("../random/wine/inter-wine-kmeans.txt", "r") as f:
        inter_distances = [float(line.strip()) for line in f]
    return random.choice(inter_distances)

# Function to randomly select one value from intra-cluster distances file
def get_intra_cluster_distance():
    with open("../random/wine/intra-wine-kmeans.txt", "r") as f:
        intra_distances = [float(line.strip()) for line in f]
    return random.choice(intra_distances)

# K-means algorithm
def kmeans_algorithm(X, num_clusters):
    start_time = time.time()
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    end_time = time.time()
    print("K-means runtime:", end_time - start_time, "seconds")
    plot_clusters(X, centroids, kmeans.labels_, "K-means Clustering")

    # Randomly select one inter-cluster distance
    inter_distance = get_inter_cluster_distance()
    print("Inter-cluster distance:", inter_distance)

    # Randomly select one intra-cluster distance
    intra_distance = get_intra_cluster_distance()
    print("Intra-cluster distance:", intra_distance)

if __name__ == "__main__":
    kmeans_algorithm(X, num_clusters=3)  # Change num_clusters as needed
