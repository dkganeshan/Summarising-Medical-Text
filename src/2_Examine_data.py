import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(PROJECT_DIR / "data" / "processed_data" / "extracted_data.csv")

print(df.head(25))

stats_findings = df['findings'].str.split().str.len().describe()
print(stats_findings)
stats_impression = df['impression'].str.split().str.len().describe()
print(stats_impression)
