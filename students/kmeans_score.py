import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import time

# Load the student data from a CSV file into a pandas DataFrame
df = pd.read_csv('student_data.csv')

# Select the features (third and fourth semester marks) for clustering
features = df[['third_sem', 'fourth_sem']]

# Define the number of clusters
k = 5

# Record the start time
start_time = time.time()

# Perform KMeans clustering
max_iterations = 100
for _ in range(max_iterations):
    # Initialize centroids randomly for each iteration
    centroids = features.sample(n=k).values
    
    # Define a function to assign each data point to the nearest centroid
    def assign_clusters(features, centroids):
        distances = np.sqrt(((features - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    # Define a function to update the centroids based on the mean of the data points assigned to each cluster
    def update_centroids(features, clusters, k):
        new_centroids = np.zeros((k, features.shape[1]))
        for i in range(k):
            new_centroids[i] = np.mean(features[clusters == i], axis=0)
        return new_centroids

    clusters = assign_clusters(features.values, centroids)
    new_centroids = update_centroids(features.values, clusters, k)
    if np.allclose(new_centroids, centroids):
        break
    centroids = new_centroids

# Print the data points with respective clusters in matrix form
for cluster_label in np.unique(clusters):
    cluster_points = features.values[clusters == cluster_label]
    print(f"Cluster {cluster_label}:")
    print(cluster_points)
    print()

# Assign cluster labels to each data point
df['Cluster'] = clusters

# Record the end time
end_time = time.time()

# Calculate and print the running time
running_time = end_time - start_time
print(f"Running time: {running_time:.4f} seconds")

# Calculate sensitivity (Silhouette score)
silhouette_avg = silhouette_score(features, df['Cluster'])
print(f"Sensitivity (Silhouette Score): {silhouette_avg:.4f}")

# Calculate inter-cluster distance
inter_cluster_distance = np.mean(np.min(np.sqrt(((centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2).sum(axis=2)), axis=1))
print(f"Inter-cluster distance: {inter_cluster_distance:.4f}")

# Calculate intra-cluster distance
intra_cluster_distance = np.mean([np.mean(np.sqrt(((features.values[clusters == i] - centroids[i])**2).sum(axis=1))) for i in range(k)])
print(f"Intra-cluster distance: {intra_cluster_distance:.4f}")

# Calculate convergence rate
convergence_rate = _ + 1
print(f"Convergence Rate: {convergence_rate} iterations")

with open('clustering_scores.txt', 'w') as file:
    file.write(f"K Means scores:\n")
    file.write(f"Running time: {running_time:.4f} seconds\n")
    file.write(f"Sensitivity (Silhouette Score): {silhouette_avg:.4f}\n")
    file.write(f"Inter-cluster distance: {inter_cluster_distance:.4f}\n")
    file.write(f"Intra-cluster distance: {intra_cluster_distance:.4f}\n")

# Visualize the clustering results using a scatter plot
plt.scatter(features['third_sem'], features['fourth_sem'], c=df['Cluster'], cmap='tab20')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='black', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Third Semester Marks')
plt.ylabel('Fourth Semester Marks')
plt.legend()
plt.show()

# Visualize robustness to outliers
outliers_df = features.copy()
outliers_df.iloc[-1] = [50, 80]  # Introduce an outlier for demonstration purposes

# Perform K-Means clustering with outliers
df['Cluster_Outliers'] = assign_clusters(outliers_df.values, centroids)

# Visualize clustering with outliers
plt.scatter(outliers_df['third_sem'], outliers_df['fourth_sem'], c=df['Cluster_Outliers'], cmap='tab20')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='black', s=200, label='Centroids')
plt.title('K-means Clustering with Outliers')
plt.xlabel('Third Semester Marks')
plt.ylabel('Fourth Semester Marks')
plt.legend()
plt.show()
