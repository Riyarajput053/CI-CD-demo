import pickle

def predict(input_data):
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model.predict([input_data])

if __name__ == "__main__":
    sample_input = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    print(predict(sample_input))
