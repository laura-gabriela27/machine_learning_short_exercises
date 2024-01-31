"""5. Utilizați PCA pentru a comprima imaginile color din setul de date Olivetti
Faces."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

# Încărcarea setului de date Olivetti Faces
faces_data = fetch_olivetti_faces()

# Extracția imaginilor și etichetelor din setul de date
images = faces_data.images
labels = faces_data.target

n_samples, height, width = images.shape  # Dimensiunile imaginilor
images_flat = images.reshape((n_samples, height * width))  # Transformarea imaginilor în vectori

# Inițializarea și antrenarea modelului PCA pentru a reduce dimensionalitatea la 50 de componente
n_components = 50
pca = PCA(n_components=n_components)
pca.fit(images_flat)

images_compressed = pca.transform(images_flat)  # Comprimarea imaginilor

images_reconstructed = pca.inverse_transform(images_compressed) # Reconstruirea imaginilor

n_rows = 4 # Vizualizarea imaginilor originale și a celor comprimate
n_cols = 5  # Vizualizarea imaginilor originale și a celor comprimate
plt.figure(figsize=(2 * n_cols, 2 * n_rows))
plt.suptitle(f"Imagini originale vs. Imagini comprimate cu PCA ({n_components} componente)")

for i in range(n_rows):
    for j in range(n_cols):
        index = i * n_cols + j
        plt.subplot(n_rows, 2 * n_cols, 2 * index + 1)
        plt.imshow(images[index], cmap=None)  # Original color image
        plt.axis('off')
        if j == 0:
            plt.title("Original")
        plt.subplot(n_rows, 2 * n_cols, 2 * index + 2)
        plt.imshow(images_reconstructed[index].reshape((height, width)), cmap=None)  # Compressed color image
        plt.axis('off')
        if j == 0:
            plt.title("Comprimat")

plt.show()



