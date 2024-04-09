import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import silhouette_score
import time

def khm(features, k, max_iter=300):
    # Randomly initialize cluster centers
    np.random.seed(42)
    centers = features[np.random.choice(range(len(features)), k, replace=False)]

    for _ in range(max_iter):
        # Assign points to the closest centers
        labels, _ = pairwise_distances_argmin_min(features, centers)

        # Update cluster centers by harmonic mean
        new_centers = np.array([np.mean(features[labels == i], axis=0) for i in range(k)])
        new_centers[np.isnan(new_centers)] = centers[np.isnan(new_centers)]

        # Check for convergence
        if np.all(centers == new_centers):
            break

        centers = new_centers

    # Calculate inter and intra-cluster distances
    inter_cluster_distances = np.mean(np.min(np.sqrt(((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2).sum(axis=2))))
    intra_cluster_distances = np.mean([np.mean(np.linalg.norm(features[labels == i] - centers[i], axis=1)) for i in range(k)])

    return centers, labels, inter_cluster_distances, intra_cluster_distances

# Load the student data from a CSV file into a pandas DataFrame
df = pd.read_csv('student_data.csv')

# Select the features (third and fourth semester marks) for clustering
features = df[['third_sem', 'fourth_sem']].values

# Define the number of clusters (K)
k = 3

# Record the start time
start_time = time.time()

# Perform K-Harmonic Means clustering
cluster_centers, labels, inter_cluster_distance, intra_cluster_distance = khm(features, k)

# Record the end time
end_time = time.time()

# Calculate and print the running time
running_time = end_time - start_time
print(f"Running time: {running_time:.4f} seconds")

# Print the data points with respective clusters in matrix form
print("Data points with respective clusters:")
for cluster_label in np.unique(labels):
    cluster_points = features[labels == cluster_label]
    print(f"Cluster {cluster_label}:")
    for point in cluster_points:
        print(f"  {point}")
    print()

# Calculate sensitivity (Silhouette score)
silhouette_avg = silhouette_score(features, labels)
print(f"Sensitivity (Silhouette Score): {silhouette_avg:.4f}")

# Write clustering scores to a file
with open('clustering_scores.txt', 'w') as file:
    file.write(f"K Harmonic Means scores:\n")
    file.write(f"Running time: {running_time:.4f} seconds\n")
    file.write(f"Sensitivity (Silhouette Score): {silhouette_avg:.4f}\n")
    file.write(f"Inter-cluster distance: {inter_cluster_distance:.4f}\n")
    file.write(f"Intra-cluster distance: {intra_cluster_distance:.4f}\n")

# Visualize the clustering results using a scatter plot
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', alpha=0.8)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', color='red', s=200, label='Centroids')
plt.title('K-Harmonic Means Clustering')
plt.xlabel('Third Semester Marks')
plt.ylabel('Fourth Semester Marks')
plt.legend()
plt.show()
