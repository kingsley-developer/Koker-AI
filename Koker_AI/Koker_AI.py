import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI

def model3():
    workers = pd.read_csv(filepath_or_buffer="Employee.csv")
    workers["EverBenched"] = np.where(workers["EverBenched"] == "Yes", 1, 0)

    features = workers.drop(["Education", "JoiningYear", "City", "PaymentTier", "Gender", "LeaveOrNot"], axis=1)
    labels = workers["LeaveOrNot"]

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    kingsley_model = RandomForestClassifier()
    kingsley_model.fit(x_train, y_train)

    predictions = kingsley_model.predict(x_test)

    mse = mean_squared_error(y_test, predictions)

    joblib.dump(kingsley_model, "kingsley_worker_model.joblib")

    loaded_model = joblib.load("kingsley_worker_model.joblib")

    y_pred_loaded = loaded_model.predict(x_test)

    accuracy_loaded = accuracy_score(y_test, y_pred_loaded)

    print("Koker_AI model accuracy: ", accuracy_loaded)
    print(f'Koker_AI model Mean Squared Error: {mse}')
    print("Koker_AI model predictions: ", predictions)
    return predictions

app = FastAPI()

@app.get("/get_ai_predictions")
def getData():
    return {"predictions" : model3()}
