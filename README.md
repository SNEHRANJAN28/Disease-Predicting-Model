# 🩺 HealthHive: AI-Powered Symptom Classifier & Diagnosis API

HealthHive is a production-ready Machine Learning API built with **FastAPI** and **Scikit-Learn** that classifies user-submitted symptoms into medical specialties, predicts the top 3 most likely conditions, identifies potential macronutrient deficiencies, and suggests clinical diagnostic tests. 

By utilizing an optimized, dynamic **Multinomial Naive Bayes** pipeline trained on over 100,000 clinical records, the system contextually isolates diagnoses to maximize prediction accuracy.

---

## 🛠️ System Architecture & Workflow

To bypass the memory and hardware limitations of hosting large-tier datasets on cloud free-tiers (like Render), HealthHive utilizes a separated dual-stage pipeline:

1. **Asset Generation (`prepare_assets.py`):** Processes the raw `100,000+` row dataset locally, fits label encoders, maps text arrays via a `TfidfVectorizer`, and serializes the objects into highly compressed binary `.joblib` deployment modules.
2. **Production Web Server (`main.py`):** A lightweight FastAPI server that mounts the serialized components directly into memory, rendering instant predictions without runtime data bottlenecks.

---

## 📦 Project Directory Structure

```text
📁 Disease-Predicting-Model/
│
├── main.py                # Core FastAPI application serving endpoints
├── requirements.txt       # Frozen production dependencies for deployment
├── tfidf.joblib           # Pre-fitted frozen text vectorizer binary
├── processed_df.joblib    # Serialized lookup arrays and encoded features
└── README.md              # Documentation

```

---

## ⚡ Tech Stack & Core Libraries

* **Backend Framework:** FastAPI (Asynchronous Python Web Framework)
* **Production Server:** Uvicorn (ASGI web server)
* **Machine Learning Pipeline:** Scikit-Learn (`MultinomialNB`, `TfidfVectorizer`, `LabelEncoder`)
* **Data Processing & Vectors:** Pandas & NumPy
* **Serialization:** Joblib

---

## 🔌 API Endpoints Reference

### 1. Service Health Check

Verify if the deployment instance is live.

* **Method:** `GET`
* **Path:** `/`
* **Response:**
```json
{
  "status": "healthy",
  "service": "HealthHive API"
}

```



### 2. Predict Patient Diagnosis

Submit patient metadata along with natural language symptom descriptions.

* **Method:** `POST`
* **Path:** `/diagnose`
* **Request Header:** `Content-Type: application/json`
* **Request Body Payload:**
```json
{
  "name": "Rahul",
  "age": 23,
  "gender": "Male",
  "symptoms": "I am experiencing severe chest pain and heart palpitations, along with a bit of fatigue"
}

```


* **Successful Response Payload (Top 3 Likely Matches):**
```json
{
  "meta": {
    "patient_name": "Rahul",
    "age": 23,
    "gender": "Male",
    "extracted_symptoms": ["chest pain", "heart palpitations", "fatigue"],
    "matched_categories": ["Cardiology"]
  },
  "predictions": [
    {
      "possible_disease": "Coronary Artery Disease",
      "category": "Cardiology",
      "prescription": "Aspirin, Beta-Blockers, lifestyle modifications",
      "macronutrient_deficiency": "Coenzyme Q10 / Magnesium",
      "suggested_tests": ["ECG", "Lipid Profile"]
    },
    {
      "possible_disease": "Hypertension",
      "category": "Cardiology",
      "prescription": "Amlodipine, lifestyle modifications",
      "macronutrient_deficiency": "Potassium",
      "suggested_tests": ["ECG", "Lipid Profile"]
    },
    {
      "possible_disease": "General Fatigue Syndrome",
      "category": "Cardiology",
      "prescription": "Rest, hydration",
      "macronutrient_deficiency": "Iron",
      "suggested_tests": ["ECG", "Lipid Profile"]
    }
  ]
}

```



---

## 💻 Local Installation & Setup

If you want to run this project locally on your machine:

1. **Clone the Repository:**
```bash
git clone [https://github.com/SNEHRANJAN28/Disease-Predicting-Model.git](https://github.com/SNEHRANJAN28/Disease-Predicting-Model.git)
cd Disease-Predicting-Model

```


2. **Install Local Requirements:**
```bash
pip install -r requirements.txt

```


3. **Spin Up the Uvicorn Server Locally:**
```bash
uvicorn main:app --reload

```



---

## 🛡️ Medical Disclaimer

HealthHive is an educational Machine Learning project built for classification purposes. It does not provide real clinical medical advice, treatment paths, or official physician diagnoses. Always consult a licensed medical professional for health evaluations.

```

```

```

```markdown
---

## 📦 Project Directory Structure

```text
📁 Disease-Predicting-Model/
│
├── main.py                # Core FastAPI application serving endpoints
├── requirements.txt       # Frozen production dependencies for deployment
├── tfidf.joblib           # Pre-fitted frozen text vectorizer binary
├── processed_df.joblib    # Serialized lookup arrays and encoded features
└── README.md              # Documentation

```

---

## ⚡ Tech Stack & Core Libraries

* **Backend Framework:** FastAPI (Asynchronous Python Web Framework)
* **Production Server:** Uvicorn (ASGI web server)
* **Machine Learning Pipeline:** Scikit-Learn (`MultinomialNB`, `TfidfVectorizer`, `LabelEncoder`)
* **Data Processing & Vectors:** Pandas & NumPy
* **Serialization:** Joblib

---

## 🔌 API Endpoints Reference

### 1. Service Health Check

Verify if the deployment instance is live.

* **Method:** `GET`
* **Path:** `/`
* **Response:**
```json
{
  "status": "healthy",
  "service": "HealthHive API"
}

```



### 2. Predict Patient Diagnosis

Submit patient metadata along with natural language symptom descriptions.

* **Method:** `POST`
* **Path:** `/diagnose`
* **Request Header:** `Content-Type: application/json`
* **Request Body Payload:**
```json
{
  "name": "Rahul",
  "age": 23,
  "gender": "Male",
  "symptoms": "I am experiencing severe chest pain and heart palpitations, along with a bit of fatigue"
}

```


* **Successful Response Payload (Top 3 Likely Matches):**
```json
{
  "meta": {
    "patient_name": "Rahul",
    "age": 23,
    "gender": "Male",
    "extracted_symptoms": ["chest pain", "heart palpitations", "fatigue"],
    "matched_categories": ["Cardiology"]
  },
  "predictions": [
    {
      "possible_disease": "Coronary Artery Disease",
      "category": "Cardiology",
      "prescription": "Aspirin, Beta-Blockers, lifestyle modifications",
      "macronutrient_deficiency": "Coenzyme Q10 / Magnesium",
      "suggested_tests": ["ECG", "Lipid Profile"]
    },
    {
      "possible_disease": "Hypertension",
      "category": "Cardiology",
      "prescription": "Amlodipine, lifestyle modifications",
      "macronutrient_deficiency": "Potassium",
      "suggested_tests": ["ECG", "Lipid Profile"]
    },
    {
      "possible_disease": "General Fatigue Syndrome",
      "category": "Cardiology",
      "prescription": "Rest, hydration",
      "macronutrient_deficiency": "Iron",
      "suggested_tests": ["ECG", "Lipid Profile"]
    }
  ]
}

```



---

## 💻 Local Installation & Setup

If you want to run this project locally on your machine:

1. **Clone the Repository:**
```bash
git clone [https://github.com/SNEHRANJAN28/Disease-Predicting-Model.git](https://github.com/SNEHRANJAN28/Disease-Predicting-Model.git)
cd Disease-Predicting-Model

```


2. **Install Local Requirements:**
```bash
pip install -r requirements.txt

```


3. **Spin Up the Uvicorn Server Locally:**
```bash
uvicorn main:app --reload

```


4. Open `http://127.0.0.1:8000/docs` in your web browser to test using the local Swagger interface.

---

## 🛡️ Medical Disclaimer

HealthHive is an educational Machine Learning project built for classification purposes. It does not provide real clinical medical advice, treatment paths, or official physician diagnoses. Always consult a licensed medical professional for health evaluations.

```

---

### Step 3: Save and Push to GitHub
1. Press `Control + O`, hit `Enter` to save the file in nano.
2. Press `Control + X` to exit.
3. Push it directly to your GitHub repo using these commands:
   ```bash
   git add README.md
   git commit -m "Add production-ready documentation README"
   git push origin main

```

Now when anyone hits your GitHub link, they will see a beautiful, clean presentation of your hard work! Let me know when the Render deployment goes completely green.
