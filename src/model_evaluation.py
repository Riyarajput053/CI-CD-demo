import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error

def evaluate_model():
    df = pd.read_csv("data/processed_data.csv")
    X = df.drop(columns=['medv'])
    y = df['medv']

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X)
    error = mean_absolute_error(y, predictions)
    print(f"Model MAE: {error}")

if __name__ == "__main__":
    evaluate_model()
