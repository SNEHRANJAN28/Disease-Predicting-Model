from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import joblib
import string
import re

# 1. Initialize FastAPI application
app = FastAPI(
    title="HealthHive ML API",
    description="Production API for symptom-based disease classification",
    version="1.0"
)

# 2. Load the pre-processed assets globally at startup
print("⏳ Loading deployment assets into memory...")
try:
    tfidf = joblib.load("tfidf.joblib")
    df = joblib.load("processed_df.joblib")
    print("🚀 Assets loaded successfully. HealthHive API is ready!")
except Exception as e:
    print(f"❌ Error loading assets: {e}")
    print("Please ensure 'tfidf.joblib' and 'processed_df.joblib' are in the same directory.")

# 3. Static Mapping Configurations
keyword_category_map = {
    "Urology": ["urination", "pee", "frequent urination", "burning sensation", "blood in urine"],
    "Neurology": ["headache", "dizzy", "numbness", "seizure", "confusion", "memory loss", "nausea"],
    "Cardiology": ["chest pain", "palpitations", "shortness of breath", "heartburn", "high blood pressure", "fatigue"],
    "Dermatology": ["rash", "itching", "skin", "blisters", "redness", "pimples", "oily skin"],
    "Pulmonology": ["cough", "wheezing", "breath", "asthma", "shortness of breath", "breathlessness"],
    "Gastroenterology": ["vomit", "diarrhea", "abdomen", "stomach pain", "constipation", "bloating", "abdominal pain"],
    "Orthopedics": ["joint pain", "bone pain", "fracture", "back pain", "joint swelling", "stiffness", "swelling", "pain on movement"],
    "Endocrinology": ["diabetes", "thyroid", "fatigue", "weight gain", "weight loss", "thirst"]
}

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

# 4. Define the expected Input JSON structure using Pydantic
class PatientRequest(BaseModel):
    name: str
    age: int
    gender: str
    symptoms: str

def preprocess_symptoms(symptoms_text):
    delimiters = [',', ';', ' and ', ' with ', ' or ', '.']
    for d in delimiters:
        symptoms_text = symptoms_text.replace(d, ',')
    symptom_list = [s.strip().lower() for s in symptoms_text.split(',') if s.strip()]
    symptom_list = [''.join(ch for ch in s if ch not in string.punctuation) for s in symptom_list]
    return symptom_list

# 5. Base Health Check Endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "service": "HealthHive API"}

# 6. Core Diagnosis Endpoint
@app.post("/diagnose")
def diagnose_patient(payload: PatientRequest):
    symptom_list = preprocess_symptoms(payload.symptoms)

    # Match categories based on symptom keywords
    matched_categories = set()
    for symptom in symptom_list:
        for cat, keywords in keyword_category_map.items():
            if any(keyword in symptom for keyword in keywords):
                matched_categories.add(cat)

    # Filter dataset based on category context or fallback to keyword regex
    if matched_categories:
        filtered_df = df[df['Category'].isin(matched_categories)]
    else:
        pattern = '|'.join(re.escape(s) for s in symptom_list)
        filtered_df = df[df['Symptoms'].str.lower().str.contains(pattern, regex=True, na=False)]

    if filtered_df.empty:
        return {"error": "No matching clinical records found for the given symptoms."}

    # Vectorize filtered context data
    X = tfidf.transform(filtered_df['Symptoms'].str.lower())

    # Fit the dynamic Naive Bayes classifiers
    disease_model = MultinomialNB().fit(X, filtered_df['Disease_Encoded'])
    prescription_model = MultinomialNB().fit(X, filtered_df['Prescription_Encoded'])
    deficiency_model = MultinomialNB().fit(X, filtered_df['Deficiency_Encoded'])

    # Transform user prediction input
    user_vec = tfidf.transform([" ".join(symptom_list)])

    # Extract top 3 disease probabilities safely
    disease_probs = disease_model.predict_proba(user_vec)[0]
    top_indices = np.argsort(disease_probs)[-3:][::-1]
    top_disease_encodings = [disease_model.classes_[i] for i in top_indices if i < len(disease_model.classes_)]

    # Map lookups efficiently
    disease_lookup = filtered_df[['Disease_Encoded', 'Disease', 'Category']].drop_duplicates().set_index('Disease_Encoded')
    prescription_lookup = filtered_df[['Disease', 'Prescription']].drop_duplicates().set_index('Disease')['Prescription'].to_dict()
    deficiency_lookup = filtered_df[['Disease', 'Macronutrient_Deficiency']].drop_duplicates().set_index('Disease')['Macronutrient_Deficiency'].to_dict()

    # Construct JSON Response array
    diagnosis_reports = []
    for enc in top_disease_encodings:
        if enc in disease_lookup.index:
            row = disease_lookup.loc[enc]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            
            disease_name = row['Disease']
            category_name = row['Category']
            
            diagnosis_reports.append({
                "possible_disease": disease_name,
                "category": category_name,
                "prescription": prescription_lookup.get(disease_name, "Consult Specialist"),
                "macronutrient_deficiency": deficiency_lookup.get(disease_name, "Unknown"),
                "suggested_tests": test_suggestions.get(category_name, ["General Blood Test"])
            })

    return {
        "meta": {
            "patient_name": payload.name,
            "age": payload.age,
            "gender": payload.gender,
            "extracted_symptoms": symptom_list,
            "matched_categories": list(matched_categories) if matched_categories else ["General"]
        },
        "predictions": diagnosis_reports
    }