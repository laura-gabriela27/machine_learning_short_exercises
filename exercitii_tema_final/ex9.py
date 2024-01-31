"""9. Utilizați setul de date digits și aplicați PCA pentru a reduce
dimensionalitatea datelor și pentru a vizualiza varianța explicată de diferite
număr de componente principale."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

digits = load_digits()  # Încărcarea setului de date digits
X = digits.data
y = digits.target

# Inițializarea și antrenarea modelului PCA
pca = PCA()
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

# Vizualizarea explicabilității varianței
exp_var_rat = pca.explained_variance_ratio_
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(exp_var_rat) + 1), np.cumsum(exp_var_rat), marker='o', linestyle='-', color='purple')
plt.title('Explained Variance Ratio')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Vizualizarea varianței explicată de diferite număr de componente principale
plt.figure(figsize=(12, 6))
plt.bar(range(1, len(exp_var_rat) + 1), exp_var_rat, align='center', alpha=0.5, color='purple')
plt.title('Explained Variance Ratio')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.grid(True)
plt.show()
