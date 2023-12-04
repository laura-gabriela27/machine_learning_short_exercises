import pandas as pd
from faker import Faker
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Utilizați un set de date Airbnb pentru a analiza prețurile și ocuparea locuințelor într-o anumită locație.
# Creați hărți de căldură pentru a evidenția zonele populare și grafice pentru a înțelege factorii care influențează
# prețurile.
# Setul de date trebuie sa aiba urmatoarea structura:
# data = {'Neighborhood': ['A', 'B', 'C', 'D', 'E'],
# 'Price': [100, 120, 80, 150, 90],
# 'Occupancy': [80, 60, 90, 50, 70],
# 'Review_Score': [4.5, 4.2, 4.8, 3.9, 4.6]}
# Creati un set de date cu 100 de inregistrari pentru structura de mai sus. In coloana 'Neighborhood' trebuie sa
# avem date duplicat.

fake = Faker()
data = {'Neighborhood': [], 'Price': [], 'Occupancy': [], 'Review_Score': []}
for i in range(100):
    data['Neighborhood'].append(random.choice(['A', 'B', 'C', 'D', 'E']))
    data['Price'].append(random.randint(50, 200))
    data['Occupancy'].append(random.randint(1, 100))
    data['Review_Score'].append(round(random.uniform(1, 5), 2))

df = pd.DataFrame(data)
print(df)

#heatmap for Ocuppancy
plt.figure(figsize=(10, 6))
heatmap_data = df.pivot_table(index='Neighborhood', values=['Price', 'Occupancy', 'Review_Score'], aggfunc='mean')
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap for Airbnb data')
plt.show()

#box graph for Price -> in this way we see both the minimum and the maximum price for a specific neighborhood
plt.figure(figsize=(10, 6))
sns.boxplot(x='Neighborhood', y='Price', data=df, palette='viridis')
plt.title('Price for Airbnb data in different neighborhoods')
plt.show()

#graph for review score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Price', y='Review_Score', data=df, hue='Neighborhood', palette='colorblind')
plt.title('Review score for Airbnb data over price in different neighborhoods')
plt.show()

#graph for occupancy in terms of price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Price', y='Occupancy', data=df, hue='Neighborhood', palette='dark')
plt.title('Occupancy for Airbnb data in terms of price in different neighborhoods')
plt.show()

#violin graph for occupancy
plt.figure(figsize=(10, 6))
sns.violinplot(x='Neighborhood', y='Occupancy', data=df, palette='pastel')
plt.title('Occupancy Distribution Across Neighborhoods')
plt.show()

#correlation matrix for occupancy, price and review score
plt.figure(figsize=(10, 6))
sns.heatmap(df.groupby('Neighborhood').mean()[['Price', 'Occupancy', 'Review_Score']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap for Airbnb Data')
plt.show()

