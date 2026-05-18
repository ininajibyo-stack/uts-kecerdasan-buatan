import pandas as pd
import glob

# Ambil semua file CSV
files = glob.glob('data/*.csv')

# Menyimpan dataframe
df_list = []

for file in files:

    print(f"Membaca: {file}")

    try:

        # Membaca CSV dengan mode aman
        df = pd.read_csv(
            file,
            encoding='utf-8',
            sep=None,
            engine='python',
            on_bad_lines='skip'
        )

        # Hapus data kosong
        df.dropna(inplace=True)

        # Tambahkan dataframe
        df_list.append(df)

        print("Berhasil dibaca\n")

    except Exception as e:

        print(f"Gagal membaca {file}")
        print("Error:", e)
        print()

# Gabungkan semua data
combined_df = pd.concat(df_list, ignore_index=True)

# Simpan hasil gabungan
combined_df.to_csv('dataset_gabungan.csv', index=False)

print("================================")
print("Dataset berhasil digabung!")
print("Jumlah data:", len(combined_df))
print("================================")