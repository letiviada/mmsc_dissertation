import os
import joblib

def save_model(model, model_path):
    model_dir = os.path.dirname(model_path)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")