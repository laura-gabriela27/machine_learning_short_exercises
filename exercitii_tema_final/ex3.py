"""3. Aplicați PCA pe setul de date Breast Cancer și vizualizați explicabilitatea
varianței."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()  # Încărcarea setului de date Breast Cancer
pca = PCA()  # Inițializarea modelului PCA

X_pca = pca.fit_transform(StandardScaler().fit_transform(data.data))  # Antrenarea modelului PCA

exp_var_rat = pca.explained_variance_ratio_  # Explicabilitatea varianței

plt.figure(figsize=(12, 6))  # Vizualizarea explicabilității varianței
plt.plot(range(1, len(exp_var_rat) + 1), np.cumsum(exp_var_rat), marker='o', linestyle='-', color='purple')
plt.title('Explained Variance Ratio')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()