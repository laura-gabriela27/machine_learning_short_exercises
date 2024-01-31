""". Utilizați setul de date iris și aplicați algoritmul K-Means pentru a grupa
datele în 3 clustere, bazându-vă pe caracteristicile 'sepal length' și 'sepal
width'."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Încărcarea setului de date Iris
iris = load_iris()
X = iris.data[:, [0, 1]]  # Selectarea doar a caracteristicilor 'sepal length' și 'sepal width'

# Inițializarea și antrenarea modelului K-Means cu 3 clustere
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Efectuarea predicțiilor pentru gruparea datelor
labels = kmeans.labels_

# Vizualizarea grupărilor pe un grafic
plt.figure(figsize=(12, 6))

for i in range(3):
    cluster_pts = X[labels == i]
    plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], label=f'Cluster {i + 1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Clustering Iris Data with K-Means')
plt.legend()
plt.show()

