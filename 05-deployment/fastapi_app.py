from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Define input data model
class ClientData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Create FastAPI app
app = FastAPI(title="Lead Conversion API", version="1.0.0")

# Load the model from the base image (pipeline_v2.bin)
with open('/pipeline_v2.bin', 'rb') as f:
    model = pickle.load(f)

@app.get("/")
async def root():
    return {"message": "Lead Conversion Prediction API"}

@app.get("/ping")
async def ping():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(client: ClientData):
    """Predict client conversion probability"""
    try:
        # Convert input data to dictionary format
        client_dict = {
            "lead_source": client.lead_source,
            "number_of_courses_viewed": client.number_of_courses_viewed,
            "annual_income": client.annual_income
        }
        
        # Make prediction
        probability = model.predict_proba([client_dict])[0][1]
        
        return {
            "lead_source": client.lead_source,
            "number_of_courses_viewed": client.number_of_courses_viewed,
            "annual_income": client.annual_income,
            "conversion_probability": round(probability, 3),
            "will_convert": probability > 0.5
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
