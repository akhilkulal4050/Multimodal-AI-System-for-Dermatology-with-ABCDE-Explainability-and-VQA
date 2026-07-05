import kagglehub
import pandas as pd
# Download latest version
path = kagglehub.dataset_download("ksabishek/massive-bank-dataset-1-million-rows")

print("Path to dataset files:", path)
df = pd.read_excel(f"{path}/bankdataset.xlsx", engine='openpyxl')
print(df.head())
print("Dataset shape:", df.shape)