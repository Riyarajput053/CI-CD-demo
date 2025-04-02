import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from model import get_model  

def train_model():
    df = pd.read_csv("data/processed_data.csv")
    X = df.drop(columns=['medv'])
    y = df['medv']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = get_model()  # Using the function from model.py
    model.fit(X_train, y_train)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model training complete and saved to models/model.pkl.")

if __name__ == "__main__":
    train_model()
