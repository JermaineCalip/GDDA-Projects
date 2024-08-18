# Import Necessary Packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Data points
df = np.array([
    [2, 10],  # A1
    [2, 5],   # A2
    [8, 4],   # A3
    [5, 8],   # A4
    [7, 5],   # A5
    [6, 4],   # A6
    [1, 2],   # A7
    [4, 9]    # A8
])

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df)

# Centroids and Labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Display the results
print("Centroids:")
print(centroids)
print("\nLabels:")
print(labels)

# Visualizations
plt.figure(figsize=(12, 8))
plt.scatter(df[:, 0], df[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
# Labels
plt.title('K-Means Clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
