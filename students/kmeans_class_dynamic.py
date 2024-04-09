import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the student data from a CSV file into a pandas DataFrame
df = pd.read_csv('student_data.csv')

# Select the features (third and fourth semester marks) for clustering
features = df[['third_sem', 'fourth_sem']]

# Define the number of clusters
k = 4

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

# Assign cluster labels to each data point
df['Cluster'] = clusters

# Visualize the clustering results using a scatter plot
plt.scatter(features['third_sem'], features['fourth_sem'], c=df['Cluster'], cmap='tab20')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='black', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Third Semester Marks')
plt.ylabel('Fourth Semester Marks')
plt.legend()
plt.show()
