import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load dataset

df = pd.read_csv('data/produksi_padi_jabar.csv')

X = df[['Luas_Panen', 'Produksi_Padi']]

# Normalisasi
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
inertia = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Inertia')
plt.show()

# Model final
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualisasi
plt.scatter(df['Luas_Panen'], df['Produksi_Padi'], c=df['Cluster'])
plt.xlabel('Luas Panen')
plt.ylabel('Produksi Padi')
plt.title('Cluster Produksi Padi')
plt.show()

print(df[['Luas_Panen', 'Produksi_Padi', 'Cluster']].head())