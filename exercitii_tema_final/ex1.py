""" 1. Folosiți setul de date Iris pentru a realiza clusterizare k-means. """

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

iris = datasets.load_iris()  # Încărcarea setului de date Iris
X = iris.data   # Extracția datelor de intrare

kmeans = KMeans(n_clusters=3, n_init=10)    # Inițializarea modelului K-Means cu 3 clustere
kmeans.fit(X)  # Antrenarea modelului K-Means

labels = kmeans.labels_ # Etichetele clusterelor
centers = kmeans.cluster_centers_  # Centroidurile clusterelor

plt.figure(figsize=(12, 6))

for i in range(3):  # Vizualizarea clusterelor
    cluster_pts = X[labels == i]
    plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], label=f'Cluster {i + 1}')

plt.scatter(centers[:, 0], centers[:, 1], c='purple', marker='X', s=200, label='Centroids')  # Vizualizarea centroidurilor

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-means - Clustering Iris Data with Cluster Centers')   # Titlul graficului
plt.legend()
plt.show()