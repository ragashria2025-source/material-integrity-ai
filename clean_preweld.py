import pandas as pd

INPUT_FILE = "dataset/WSteel_Dataset.csv"
OUTPUT_FILE = "dataset/WSteel_Preweld_Clean.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE, sep=";")

print("Original shape:", df.shape)

# Clean column names
df.columns = df.columns.str.strip()

# Convert decimal-comma values to numbers
numeric_columns = [
    "welding speed (m/min)",
    "material strenght steel (mm)",
]

for column in numeric_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

# Convert target to 0/1
df["cracking in the weld metal (yes/no)"] = (
    df["cracking in the weld metal (yes/no)"]
    .str.strip()
    .str.lower()
    .map({"no": 0, "yes": 1})
)

# Keep only potential pre-weld variables
features = [
    "power (W)",
    "welding speed (m/min)",
    "gas flow rate (l/min)",
    "focal position (mm)",
    "angular position (�)",
    "material strenght steel (mm)",
]

target = "cracking in the weld metal (yes/no)"

df_clean = df[features + [target]].copy()

print("\nCleaned shape:", df_clean.shape)

print("\nCleaned data:")
print(df_clean.head(10).to_string())

print("\nData types:")
print(df_clean.dtypes)

print("\nMissing values:")
print(df_clean.isnull().sum())

print("\nTarget distribution:")
print(df_clean[target].value_counts())

# Save cleaned dataset
df_clean.to_csv(OUTPUT_FILE, index=False)

print("\nSaved cleaned dataset to:")
print(OUTPUT_FILE)