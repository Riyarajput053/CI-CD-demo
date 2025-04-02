import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    df = pd.read_csv("data/raw_data.csv")
    scaler = StandardScaler()
    X = df.drop(columns=['medv'])
    X_scaled = scaler.fit_transform(X)
    y = df['medv']
    
    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df["medv"] = y
    processed_df.to_csv("data/processed_data.csv", index=False)
    print("Data preprocessing complete.")

if __name__ == "__main__":
    preprocess_data()
