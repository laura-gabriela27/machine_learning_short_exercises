"""2. Utilizați K-means pentru a grupa clienții în funcție de cheltuielile lor
într-un set de date despre comerțul cu ridicata (Wholesale
customers).Setul de date se preia de la
https://archive.ics.uci.edu/ml/machine-learning-databa
ses/00292/Wholesale%20customers%20data.csv """

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("data.csv")

cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = data[cols]

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)  # Inițializarea modelului K-Means cu 6 clustere
data['Cluster'] = kmeans.fit_predict(StandardScaler().fit_transform(X))  # Antrenarea modelului K-Means

for cluster in range(6):  # Vizualizarea clusterelor
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['Fresh'], cluster_data['Frozen'], label=f'Cluster {cluster + 1}')

plt.scatter(kmeans.cluster_centers_[:, cols.index('Fresh')],  # Vizualizarea centroidurilor
            kmeans.cluster_centers_[:, cols.index('Frozen')],
            s=300, c='purple', marker='X', label='Centroids')

plt.xlabel('Fresh')  # Titlul axei x
plt.ylabel('Frozen')  # Titlul axei y
plt.legend()
plt.show()