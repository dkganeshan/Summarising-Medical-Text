from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split # scikit-learn to be installed in the environment


PROJECT_DIR = Path(__file__).resolve().parent.parent
df_reports = pd.read_csv(PROJECT_DIR / "data" / "processed_data" / "extracted_data.csv")

# Split the dataset
train_df, test_df = train_test_split(
    df_reports,
    test_size=0.15,
    random_state=42
)

print("Training set:", train_df.shape)
print("Test set:", test_df.shape)

train_df.to_csv(PROJECT_DIR / "data" / "processed_data" / "train_data.csv", index=False)
test_df.to_csv(PROJECT_DIR / "data" / "processed_data" / "test_data.csv", index=False)

print(f"Training records: {len(train_df)}")
print(f"Testing records: {len(test_df)}")