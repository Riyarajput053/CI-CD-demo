import pandas as pd
import os

# Dataset URL (Boston Housing Dataset)
DATA_URL = "https://raw.githubusercontent.com/dataprofessor/data/master/BostonHousing.csv"

def data_ingestion():
    # Create data directory if it doesn’t exist
    os.makedirs("data", exist_ok=True)

    # Load the dataset
    df = pd.read_csv(DATA_URL)

    # Save the dataset locally
    df.to_csv("data/raw_data.csv", index=False)

    print("✅ Data ingestion complete. Dataset saved to 'data/raw_data.csv'.")

if __name__ == "__main__":
    data_ingestion()
