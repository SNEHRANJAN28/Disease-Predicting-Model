from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import joblib
import string
import re

app = FastAPI(
    title="HealthHive ML API",
    description="Production API for symptom-based disease classification",
    version="1.1"
)

print("⏳ Loading deployment assets into memory...")
try:
    tfidf = joblib.load("tfidf.joblib")
    df = joblib.load("processed_df.joblib")
    print("🚀 Assets loaded successfully. HealthHive API is ready!")
except Exception as e:
    print(f"❌ Error loading assets: {e}")

# Expanded keywords to catch variations like headache/headaches, fever, etc.
keyword_category_map = {
    "Urology": ["urination", "pee", "frequent urination", "burning sensation", "blood in urine"],
    "Neurology": ["headache", "headaches", "migraine", "dizzy", "dizziness", "numbness", "seizure", "confusion", "nausea"],
    "Cardiology": ["chest pain", "palpitations", "shortness of breath", "heartburn", "high blood pressure", "fatigue"],
    "Dermatology": ["rash", "itching", "skin", "blisters", "redness", "pimples", "oily skin"],
    "Pulmonology": ["cough", "wheezing", "breath", "asthma", "shortness of breath", "breathlessness"],
    "Gastroenterology": ["vomit", "vomiting", "diarrhea", "abdomen", "stomach pain", "constipation", "bloating"],
    "Orthopedics": ["joint pain", "bone pain", "fracture", "back pain", "stiffness", "swelling"],
    "Endocrinology": ["diabetes", "thyroid", "fatigue", "weight gain", "weight loss", "thirst", "fever", "fevers"]
}

test_suggestions = {
    "Cardiology": ["ECG", "Lipid Profile"],
    "Neurology": ["EEG", "MRI Brain"],
    "Endocrinology": ["Blood Sugar", "Thyroid Profile", "Complete Blood Count (CBC)"],
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

def preprocess_symptoms(symptoms_text):
    delimiters = [',', ';', ' and ', ' with ', ' or ', '.']
    for d in delimiters:
        symptoms_text = symptoms_text.replace(d, ',')
    symptom_list = [s.strip().lower() for s in symptoms_text.split(',') if s.strip()]
    symptom_list = [''.join(ch for ch in s if ch not in string.punctuation) for s in symptom_list]
    return symptom_list

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "HealthHive API"}

@app.post("/diagnose")
def diagnose_patient(payload: PatientRequest):
    symptom_list = preprocess_symptoms(payload.symptoms)
    search_string = " ".join(symptom_list)

    # 1. Look for explicit category keywords
    matched_categories = set()
    for symptom in symptom_list:
        for cat, keywords in keyword_category_map.items():
            if any(keyword in symptom for keyword in keywords):
                matched_categories.add(cat)

    # 2. Dynamic Safe Filtering
    if matched_categories:
        filtered_df = df[df['Category'].isin(matched_categories)]
    else:
        # Fallback: search for any partial matches in the text dataset
        pattern = '|'.join(re.escape(s) for s in symptom_list)
        filtered_df = df[df['Symptoms'].str.lower().str.contains(pattern, regex=True, na=False)]

    # 3. Ultimate Safety Fallback: If filtering returns nothing, use the ENTIRE dataset
    # This guarantees that common words like 'fever' will always find a match!
    if filtered_df.empty or len(filtered_df) < 5:
        filtered_df = df

    try:
        # Transform using the vectorizer
        X = tfidf.transform(filtered_df['Symptoms'].str.lower())

        # Fit models dynamically on the available slice safely
        disease_model = MultinomialNB().fit(X, filtered_df['Disease_Encoded'])
        prescription_model = MultinomialNB().fit(X, filtered_df['Prescription_Encoded'])
        deficiency_model = MultinomialNB().fit(X, filtered_df['Deficiency_Encoded'])

        # Predict for user input
        user_vec = tfidf.transform([search_string])
        disease_probs = disease_model.predict_proba(user_vec)[0]
        
        # Pull top 3 matches safely
        top_indices = np.argsort(disease_probs)[-3:][::-1]
        top_disease_encodings = [disease_model.classes_[i] for i in top_indices if i < len(disease_model.classes_)]

        # Global dictionary mappings to prevent drop-slice KeyError exceptions
        disease_lookup = df[['Disease_Encoded', 'Disease', 'Category']].drop_duplicates().set_index('Disease_Encoded')
        prescription_lookup = df[['Disease', 'Prescription']].drop_duplicates().set_index('Disease')['Prescription'].to_dict()
        deficiency_lookup = df[['Disease', 'Macronutrient_Deficiency']].drop_duplicates().set_index('Disease')['Macronutrient_Deficiency'].to_dict()

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
                    "macronutrient_deficiency": deficiency_lookup.get(disease_name, "Dynamic Evaluation Required"),
                    "suggested_tests": test_suggestions.get(category_name, ["General Blood Workup"])
                })

        return {
            "meta": {
                "patient_name": payload.name,
                "age": payload.age,
                "gender": payload.gender,
                "extracted_symptoms": symptom_list,
                "matched_categories": list(matched_categories) if matched_categories else ["General Medicine"]
            },
            "predictions": diagnosis_reports
        }

    except Exception as runtime_error:
        return {
            "error": "Internal processing exception handled smoothly.",
            "details": str(runtime_error)
        }
