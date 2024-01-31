"""11. Folosiți setul de date digits și aplicați K-Means pentru a grupa datele în
10 clustere, iar apoi aplicați PCA pentru a vizualiza datele într-un spațiu
2D."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns

digits = load_digits() # Încărcarea setului de date digits
data, target = digits.data, digits.target # Extracția datelor de intrare și a etichetelor

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10) # Inițializarea modelului K-Means cu 10 clustere
clusters = kmeans.fit_predict(data) # Antrenarea modelului K-Means

pca = PCA(n_components=2) # Inițializarea modelului PCA cu 2 componente
reduced_data = pca.fit_transform(data) # Antrenarea modelului PCA

result_df = pd.DataFrame({'PC1': reduced_data[:, 0], 'PC2': reduced_data[:, 1], 'Cluster': clusters, 'Target': target}) # Crearea unui DataFrame cu datele reduse dimensional
print(result_df) # Afișarea DataFrame-ului

plt.figure(figsize=(12, 6)) # Vizualizarea datelor reduse dimensional
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=result_df, palette='viridis')
plt.title('Explained Variance Ratio')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()