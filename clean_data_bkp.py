import pandas as pd
import numpy as np

print("Loading dataset...")

df = pd.read_csv("data\AguaH.csv")

print("Original shape:", df.shape)

# select monthly consumption columns
month_cols = [col for col in df.columns if "f.1_" in col]

# keep only useful metadata
meta_cols = ["USO2013", "TU"]

meta_cols = [col for col in meta_cols if col in df.columns]

df_small = df[meta_cols + month_cols]

print("Selected columns:", len(df_small.columns))

# convert wide format → long format
df_long = df_small.melt(
    id_vars=meta_cols,
    var_name="month",
    value_name="water_consumption"
)

# remove missing values
df_long = df_long.dropna()

# convert consumption to numeric
df_long["water_consumption"] = pd.to_numeric(
    df_long["water_consumption"], errors="coerce"
)

df_long = df_long.dropna()

# create synthetic features
df_long["temperature"] = np.random.randint(20, 40, size=len(df_long))
df_long["rainfall"] = np.random.randint(0, 100, size=len(df_long))
df_long["population"] = np.random.randint(10000, 50000, size=len(df_long))

# reset index
df_long = df_long.reset_index(drop=True)

# save clean dataset
df_long.to_csv("clean_water_data.csv", index=False)

print("Clean dataset created!")
print("New shape:", df_long.shape)
print(df_long.head())