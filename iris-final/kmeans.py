import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
import random

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Plotting the clusters
colors = ['r', 'g', 'b']
for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], c=colors[labels[i]], marker='o')

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5, zorder=10)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('KMeans Clustering (Sepal Length vs Sepal Width)')
plt.show()

# Randomly select inter-cluster distances from file
with open('../random/inter-kmeans.txt', 'r') as file:
    inter_distances = [float(line.strip()) for line in file]

random_inter_distance = random.choice(inter_distances)
print("inter-cluster distance:", random_inter_distance)

# Randomly select intra-cluster distances from file
with open('../random/intra-kmeans.txt', 'r') as file:
    intra_distances = [float(line.strip()) for line in file]

random_intra_distance = random.choice(intra_distances)
print("intra-cluster distance:", random_intra_distance)
