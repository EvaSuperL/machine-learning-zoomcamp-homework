import pickle
import numpy as np

def load_and_predict():
    # Load the trained pipeline
    with open('pipeline_v1.bin', 'rb') as f:
        pipeline = pickle.load(f)
    
    # The record to score
    record = {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }
    
    # Convert single record to list format for prediction
    records = [record]
    
    # Get prediction probabilities
    probabilities = pipeline.predict_proba(records)
    
    # The probability of conversion (class 1)
    conversion_probability = probabilities[0][1]
    
    print(f"Prediction probabilities: {probabilities[0]}")
    print(f"Probability that this lead will convert: {conversion_probability:.3f}")
    
    return conversion_probability

if __name__ == '__main__':
    prob = load_and_predict()
    
    # Compare with the given options
    options = [0.333, 0.533, 0.733, 0.933]
    closest_option = min(options, key=lambda x: abs(x - prob))
    
    print(f"\nClosest option: {closest_option}")
