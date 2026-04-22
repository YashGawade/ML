# -*- coding: utf-8 -*-
"""Assignment_no13.ipynb

13.  Implement K-Means clustering on Iris.csv dataset.
Determine the number of clusters using the elbow method. 
Dataset Link: https://www.kaggle.com/datasets/uciml/iris

**Import Libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

"""**Load Dataset**"""

df = pd.read_csv('/content/Iris.csv')
df.head()

"""**Understand Dataset**"""

print(df.columns)
print(df.info())

"""**Data Preprocessing**"""

# Remove unnecessary column
df = df.drop(columns=['Id'])

# Use only numerical features (for clustering)
X = df[['SepalLengthCm', 'SepalWidthCm',
        'PetalLengthCm', 'PetalWidthCm']]

"""**Feature Scaling**"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""**lbow Method**"""

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

"""**Apply K-Means**"""

kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

"""**Visualization**"""

plt.scatter(X_scaled[:, 2], X_scaled[:, 3], c=y_kmeans)
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-Means Clustering (Iris Dataset)")
plt.show()

"""**Add Cluster Label**"""

df['Cluster'] = y_kmeans
df.head()

"""**Compare with Actual Species**"""

print(pd.crosstab(df['Cluster'], df['Species']))