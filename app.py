import pickle
import numpy as np
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

# Initialize FastAPI app
app = FastAPI()

# Load trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Styled Homepage
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Model API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                background-color: #f4f4f4;
                margin: 50px;
            }
            h1 {
                color: #333;
            }
            p {
                font-size: 18px;
            }
            form {
                margin-top: 20px;
            }
            input, button {
                padding: 10px;
                font-size: 16px;
                margin: 5px;
            }
            button {
                cursor: pointer;
                background-color: #28a745;
                color: white;
                border: none;
            }
            button:hover {
                background-color: #218838;
            }
        </style>
    </head>
    <body>
        <h1>🚀 Welcome to the ML Model API</h1>
        <p>This API allows you to get predictions from a trained ML model.</p>
        <p>Use the <a href="/docs">API Documentation</a> or enter values below.</p>
        
        <form action="/predict_form" method="post">
            <input type="text" name="features" placeholder="Enter comma-separated values" required>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    """

# JSON Prediction Endpoint
@app.post("/predict/")
def predict(features: list):
    """
    Expects input as a list of numerical features.
    Example request:
    {
        "features": [0.1, 0.2, 0.3, ..., 1.3]
    }
    """
    features_array = np.array([features])
    prediction = model.predict(features_array)
    return {"prediction": prediction.tolist()}

# Form-based Prediction (For UI)
@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(features: str = Form(...)):
    feature_list = list(map(float, features.split(",")))
    features_array = np.array([feature_list])
    prediction = model.predict(features_array)

    return f"""
    <html>
    <body style="text-align: center; font-family: Arial;">
        <h2>Prediction Result</h2>
        <p>Input Features: {feature_list}</p>
        <h3>Predicted Value: {prediction.tolist()}</h3>
        <br><a href="/">Go Back</a>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
