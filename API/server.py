from contextlib import asynccontextmanager
import os
import pandas as pd
from fastapi import FastAPI
import mlflow
from pydantic import BaseModel
import uvicorn

class InputPatient(BaseModel):
    age: int
    sex: int
    cp : int
    trestbps: int
    chol : int
    fbs : int
    restecg: int
    thalach : int
    exang : int
    oldpeak : float
    slope : int
    ca : int
    thal : int    

model_uri = "models:/heart_disease_prediction/1"
mlflow_uri = os.environ["MLFLOW_TRACKING_URI"]

@asynccontextmanager                    # wrapper to manage the lifespan of the app
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(mlflow_uri)
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    app.state.model = loaded_model      # store the model in the app state, accessible via app.state.model  
    print("Model loaded in memory, server is ready to serve requests.")
    yield                               # everything before yield is startup, everything after is shutdown
    del app.state.model                 # free memory
    print("Serveur arrêté proprement")



app = FastAPI(lifespan=lifespan)


@app.get('/health', status_code=200)
async def health_check():
    return {'healthy': 'true'}

@app.get('/')
def welcome_message():
    return {'message': "FastAPI server is up."}

@app.post('/predict')
def predict (data: InputPatient):
    

    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

    features = pd.DataFrame([[
            data.age, data.sex, data.cp, data.trestbps,
            data.chol, data.fbs, data.restecg, data.thalach,
            data.exang, data.oldpeak, data.slope, data.ca, data.thal
        ]], columns=columns)
    
    y_pred = app.state.model.predict(features)
    return {'prediction': int(y_pred[0])}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)