import pandas as pd

df = pd.read_csv(r"D:\drdo\sipri_full.csv")

# ... replace karo 0 se - missing data hai
df = df.replace(0, None)

# Sirf rows lo jahan kam se kam India ya China ka data ho
df_clean = df.dropna(subset=['India', 'China'], how='all')

print(df_clean[['Year','India','Pakistan','China']].to_string())