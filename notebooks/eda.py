import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(
    'data/produktivitas_padi_di_kota_tasikmalaya.csv'
)

# Cleaning
df['produktivitas_padi'] = (
    df['produktivitas_padi']
    .astype(str)
    .str.replace(', TON/HEKTAR', '', regex=False)
)

df['produktivitas_padi'] = pd.to_numeric(
    df['produktivitas_padi'],
    errors='coerce'
)

df.dropna(inplace=True)

# Informasi dataset
print("\n=== HEAD ===")
print(df.head())

print("\n=== INFO ===")
print(df.info())

print("\n=== DESKRIPSI ===")
print(df.describe())

print("\n=== MISSING VALUE ===")
print(df.isnull().sum())

# Korelasi
plt.figure(figsize=(10,6))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap='viridis'
)

plt.title('Korelasi Dataset Produktivitas Padi')
plt.tight_layout()
plt.show()

# Histogram
plt.figure(figsize=(10,6))

plt.hist(
    df['produktivitas_padi'],
    bins=20
)

plt.title('Distribusi Produktivitas Padi')
plt.xlabel('Produktivitas')
plt.ylabel('Frekuensi')

plt.tight_layout()
plt.show()

# Trend tahunan
df_group = df.groupby('tahun')[
    'produktivitas_padi'
].mean()

plt.figure(figsize=(12,6))

plt.plot(
    df_group.index,
    df_group.values,
    marker='o'
)

plt.title('Trend Produktivitas Padi per Tahun')
plt.xlabel('Tahun')
plt.ylabel('Produktivitas')

plt.grid(True)
plt.tight_layout()
plt.show()