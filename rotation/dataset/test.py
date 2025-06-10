import pandas as pd

# Load your CSV
df = pd.read_csv('./angles.csv')

# Drop duplicates based on the 'filename' column, keeping the first occurrence
df_unique = df.drop_duplicates(subset='filename', keep='first')

# Optionally save the result
df_unique.to_csv('./angles.csv', index=False)
