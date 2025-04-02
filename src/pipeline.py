import os
from data_ingestion import data_ingestion
from data_preprocessing import preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model

def run_pipeline():
    print("🚀 Starting the ML pipeline...")

    # Create required directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Execute each pipeline step
    print("\n🔹 Step 1: Data Ingestion")
    data_ingestion()

    print("\n🔹 Step 2: Data Preprocessing")
    preprocess_data()

    print("\n🔹 Step 3: Model Training")
    train_model()

    print("\n🔹 Step 4: Model Evaluation")
    evaluate_model()

    print("\n✅ ML pipeline execution completed!")

if __name__ == "__main__":
    run_pipeline()
