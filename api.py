from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware 
import joblib 
import pandas as pd 


app = FastAPI()

# Allowing the HTML frontend to call the API 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup 
models = joblib.load("saved_models/trained_models.pkl")
feature_cols = joblib.load("saved_models/feature_columns.pkl")
df_feat = joblib.load("saved_models/last_features.pkl")

@app.get("/predict")
def predict():
    latest = df_feat[feature_cols].dropna().iloc[[-1]]
    last_date = df_feat.index[-1].date()
    
    results = {}
    for name, model in models.items():
        prob = model.predict_proba(latest)[0,1]
        results[name] = {
            "direction" : "UP" if prob >= 0.5 else "DOWN",
            "confidence" : float(round(prob * 100, 2))
            
        }
        
    return {
        "as_of_date" : str(last_date),
        "predictions": results
    }
    
@app.get("/health")
def health():
    return {"status": "ok"}
