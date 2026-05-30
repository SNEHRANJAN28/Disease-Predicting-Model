from fastapi import FastAPI
print("APP STARTING...")
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
import string

app = FastAPI(title="HEALTHHIVE API", version="1.0.0")

df = pd.read_csv("patient_dataset_100k_updated(1).csv")
print("CSV LOADED")

le_category = LabelEncoder()
df["Category_Encoded"] = le_category.fit_transform(df["Category"])
df["Disease_Encoded"] = LabelEncoder().fit_transform(df["Disease"])
df["Prescription_Encoded"] = LabelEncoder().fit_transform(df["Prescription"])
df["Deficiency_Encoded"] = LabelEncoder().fit_transform(df["Macronutrient_Deficiency"])

tfidf = TfidfVectorizer()
tfidf.fit(df["Symptoms"].str.lower())

class PatientRequest(BaseModel):
    name: str
    age: int
    gender: str
    symptoms: str

def preprocess_symptoms(symptoms_text):
    delimiters = [",", ";", " and ", " with ", " or ", "."]
    for d in delimiters:
        symptoms_text = symptoms_text.replace(d, ",")

    symptom_list = [s.strip().lower() for s in symptoms_text.split(",") if s.strip()]
    symptom_list = [
        "".join(ch for ch in s if ch not in string.punctuation)
        for s in symptom_list
    ]
    return symptom_list

def diagnose_patient(name, age, gender, symptoms):
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

    symptom_list = preprocess_symptoms(symptoms)

    if not symptom_list:
        return {"error": "No symptoms provided"}

    matched_categories = set()

    for symptom in symptom_list:
        for cat, keywords in keyword_category_map.items():
            if any(keyword in symptom for keyword in keywords):
                matched_categories.add(cat)

    if matched_categories:
        filtered_df = df[df["Category"].isin(matched_categories)]
    else:
        pattern = "|".join(re.escape(s) for s in symptom_list)
        filtered_df = df[df["Symptoms"].str.lower().str.contains(pattern, regex=True)]

    if filtered_df.empty:
        return {"message": "No matching records found for your symptoms."}

    X = tfidf.transform(filtered_df["Symptoms"].str.lower())

    y_disease = filtered_df["Disease_Encoded"]
    y_prescription = filtered_df["Prescription_Encoded"]
    y_deficiency = filtered_df["Deficiency_Encoded"]

    disease_model = MultinomialNB().fit(X, y_disease)
    prescription_model = MultinomialNB().fit(X, y_prescription)
    deficiency_model = MultinomialNB().fit(X, y_deficiency)

    symptoms_vec = tfidf.transform([" ".join(symptom_list)])

    disease_probs = disease_model.predict_proba(symptoms_vec)[0]

    top_indices = np.argsort(disease_probs)[-3:][::-1]
    top_disease_encodings = [disease_model.classes_[i] for i in top_indices]

    disease_lookup = (
        filtered_df[["Disease_Encoded", "Disease", "Category"]]
        .drop_duplicates()
        .set_index("Disease_Encoded")
    )

    prescription_lookup = (
        filtered_df[["Disease", "Prescription"]]
        .drop_duplicates()
        .set_index("Disease")["Prescription"]
        .to_dict()
    )

    deficiency_lookup = (
        filtered_df[["Disease", "Macronutrient_Deficiency"]]
        .drop_duplicates()
        .set_index("Disease")["Macronutrient_Deficiency"]
        .to_dict()
    )

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

    predictions = []

    for enc in top_disease_encodings:
        row = disease_lookup.loc[enc]
        disease = row["Disease"]
        category = row["Category"]

        predictions.append({
            "disease": disease,
            "category": category,
            "prescription": prescription_lookup.get(disease, "Consult Specialist"),
            "macronutrient_deficiency": deficiency_lookup.get(disease, "Unknown"),
            "suggested_tests": test_suggestions.get(category, ["General Blood Test"])
        })

    return {
        "name": name,
        "age": age,
        "gender": gender,
        "symptoms": symptom_list,
        "matched_categories": list(matched_categories),
        "predictions": predictions
    }

@app.get("/")
def home():
    return {"message": "HEALTHHIVE API is running"}

@app.post("/predict")
def predict(patient: PatientRequest):
    return diagnose_patient(
        patient.name,
        patient.age,
        patient.gender,
        patient.symptoms
    )
