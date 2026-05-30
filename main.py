from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import string

app = FastAPI(title="HealthHive ML API", version="3.0")

print("⏳ Loading real-text production weights...")
try:
    tfidf = joblib.load("tfidf.joblib")
    disease_model = joblib.load("disease_model.joblib")
    metadata = joblib.load("metadata.joblib")
    print("🚀 Model successfully connected to memory!")
except Exception as e:
    print(f"❌ Core asset initialization failure: {e}")

test_suggestions = {
    "Cardiology": ["ECG", "Lipid Profile"],
    "Neurology": ["EEG", "MRI Brain"],
    "Endocrinology": ["Blood Sugar", "Thyroid Profile"],
    "Gastroenterology": ["Endoscopy", "Liver Function Test"],
    "Orthopedics": ["X-Ray", "Bone Density Test"],
    "Dermatology": ["Skin Biopsy"],
    "Pulmonology": ["Chest X-Ray", "Spirometry"],
    "Urology": ["Urine Analysis", "Ultrasound KUB"]
}

class PatientRequest(BaseModel):
    name: str
    age: int
    gender: str
    symptoms: str

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "HealthHive Core Engine v3"}

@app.post("/diagnose")
def diagnose_patient(payload: PatientRequest):
    try:
        clean_input = payload.symptoms.lower().translate(str.maketrans('', '', string.punctuation))
        user_vec = tfidf.transform([clean_input])
        
        # Pull probabilities directly from the text labels
        probabilities = disease_model.predict_proba(user_vec)[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]
        
        model_classes = disease_model.classes_

        diagnosis_reports = []
        for idx in top_indices:
            disease_name = str(model_classes[idx])
            category_name = metadata["category"].get(disease_name, "General Medicine")
            
            diagnosis_reports.append({
                "possible_disease": disease_name,
                "category": category_name,
                "prescription": metadata["prescription"].get(disease_name, "Consult Specialist"),
                "macronutrient_deficiency": metadata["deficiency"].get(disease_name, "Evaluation required"),
                "suggested_tests": test_suggestions.get(category_name, ["Complete Blood Count (CBC)"])
            })

        return {
            "meta": {
                "patient_name": payload.name,
                "age": payload.age,
                "gender": payload.gender,
                "input_symptoms": payload.symptoms
            },
            "predictions": diagnosis_reports
        }

    except Exception as runtime_error:
        return {
            "error": "Pipeline computation failed.",
            "details": str(runtime_error)
        }
