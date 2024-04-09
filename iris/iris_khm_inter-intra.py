import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt

def k_harmonic_means(data, k, max_iter=100, tol=1e-4):
    # Randomly initialize cluster centers
    centers = data[np.random.choice(len(data), k, replace=False)]

    for _ in range(max_iter):
        # Assign each point to the nearest cluster
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centers, axis=2), axis=1)

        # Update cluster centers using harmonic mean
        new_centers = np.array([np.mean(data[labels == i], axis=0) for i in range(k)])
        new_centers[np.isnan(new_centers)] = centers[np.isnan(new_centers)]

        # Check for convergence
        if np.linalg.norm(new_centers - centers) < tol:
            break

        centers = new_centers

    return labels, centers
 
def inter_cluster_distance(centers):
    return np.mean(np.linalg.norm(centers[:, np.newaxis] - centers, axis=2)) / centers.shape[1]

def intra_cluster_distance(data, labels, centers):
    distances = [np.mean(np.linalg.norm(data[labels == i] - centers[i], axis=1)) for i in range(len(centers))]
    return np.mean(distances) / data.shape[1]

# Read data from CSV file
iris_df = pd.read_csv('Iris.csv')

# Extract features from the dataframe
data = iris_df.iloc[:, :-1].values

# Convert target labels to numerical values
label_encoder = LabelEncoder()
iris_df['Species'] = label_encoder.fit_transform(iris_df['Species'])

# Specify the number of clusters (k)
k = 3

# Measure the running time
start_time = time.time()

# Run K-Harmonic Means algorithm
labels, centers = k_harmonic_means(data, k)

# Print the running time
end_time = time.time()
running_time = end_time - start_time
print(f'Running Time: {running_time} seconds')

# Calculate and print inter-cluster distance and intra-cluster distance
inter_dist = inter_cluster_distance(centers)
intra_dist = intra_cluster_distance(data, labels, centers)
print(f'Inter-Cluster Distance: {inter_dist}')
print(f'Intra-Cluster Distance: {intra_dist}')

# Visualize clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolors='k', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')
plt.title(f'K-Harmonic Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()