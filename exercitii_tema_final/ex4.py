"""4. Realizați reducerea dimensionalității cu PCA pe setul de date MNIST.
Cum preluam setul de date mnist = fetch_openml('mnist_784')
data = mnist.data.astype(float)
labels = mnist.target.astype(int)"""


from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Încărcarea setului de date MNIST cu parser='auto'
mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto') # am folosit si auto parser pt ca primeam un avertisment

data = mnist.data.astype(float) # Extracția datelor de intrare
labels = mnist.target.astype(int)  # Extracția etichetelor

pca = PCA(n_components=2)  # Inițializarea modelului PCA cu 2 componente
reduced_data = pca.fit_transform(data)  # Antrenarea modelului PCA

# Afișarea datelor reduse dimensional pe un grafic
plt.figure(figsize=(12, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=10)
plt.title('Explained Variance Ratio')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()


