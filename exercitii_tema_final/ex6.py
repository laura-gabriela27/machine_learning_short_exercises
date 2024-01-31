"""6. Aplicați PCA pe setul de date Glass pentru a evidenția caracteristicile
principale ale compoziției chimice a geamului. Generati 100 de randuri sub
forma de mai jos.
Glass.csv
Id,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe,Type
1,1.52101,13.64,4.49,1.10,71.78,0.06,8.75,0.00,0.00,1
2,1.51761,13.89,3.60,1.36,72.73,0.48,7.83,0.00,0.00,1
3,1.51618,13.53,3.55,1.54,72.99,0.39,7.78,0.00,0.00,1
4,1.51766,13.21,3.69,1.29,72.61,0.57,8.22,0.00,0.00,1
5,1.51742,13.27,3.62,1.24,73.08,0.55,8.07,0.00,0.00,1"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42) # Setăm seed-ul pentru reproducibilitate
num_rows = 100 # Numărul de rânduri din setul de date

# Generăm valorile pentru fiecare caracteristică folosind o distribuție normală
data = {
    'Id': np.arange(1, num_rows + 1),
    'RI': np.random.normal(loc=1.52, scale=0.01, size=num_rows),
    'Na': np.random.normal(loc=13.5, scale=0.5, size=num_rows),
    'Mg': np.random.normal(loc=4.0, scale=0.5, size=num_rows),
    'Al': np.random.normal(loc=1.3, scale=0.2, size=num_rows),
    'Si': np.random.normal(loc=72.5, scale=0.5, size=num_rows),
    'K': np.random.normal(loc=0.5, scale=0.1, size=num_rows),
    'Ca': np.random.normal(loc=8.0, scale=0.5, size=num_rows),
    'Ba': np.random.normal(loc=0.0, scale=0.1, size=num_rows),
    'Fe': np.random.normal(loc=0.0, scale=0.1, size=num_rows),
    'Type': np.random.randint(1, 8, size=num_rows)  # Simulăm tipurile de sticlă (valori între 1 și 7)
}

glass_df = pd.DataFrame(data)  # Transformăm dicționarul într-un DataFrame
glass_df.to_csv('Glass.csv', index=False) # Salvăm DataFrame-ul într-un fișier CSV
print(glass_df.head()) # Afișăm primele 5 rânduri din DataFrame

glass_df.drop('Id', axis=1, inplace=True) # Eliminăm coloana Id

# Standardizăm datele
scaler = StandardScaler()
scaled_data = scaler.fit_transform(glass_df)

# Inițializăm PCA și aplicăm pe datele standardizate
pca = PCA()
principal_components = pca.fit_transform(scaled_data)

# Afișăm varianța explicată de fiecare componentă principală
explained_variance_ratio = pca.explained_variance_ratio_
print("Varianța explicată de fiecare componentă principală:")
for i, ratio in enumerate(explained_variance_ratio, 1):
    print(f"Componenta principală {i}: {ratio:.2f}")

# Vizualizăm varianța explicată acumulată în funcție de numărul de componente
cumulative_explained_variance_ratio = explained_variance_ratio.cumsum()
plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, marker='o', linestyle='-', color='purple')
plt.xlabel('Numărul de componente principale')
plt.ylabel('Varianța explicată acumulată')
plt.title('Varianța explicată acumulată de componente principale')
plt.grid(True)
plt.show()

# Vizualizăm cele două componente principale pe un scatter plot
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=glass_df['Type'], cmap='viridis', s=10)
plt.xlabel('Componenta principală 1')
plt.ylabel('Componenta principală 2')
plt.title('PCA pe setul de date Glass')
plt.grid(True)
plt.show()