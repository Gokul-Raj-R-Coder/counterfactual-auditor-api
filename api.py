from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from google import genai
import os

app = FastAPI(title="Counterfactual Auditor API (Cloud Production)")

# Best practice for Cloud: Pull the key from an environment variable, 
# but hardcoding is fine for a quick hackathon demo.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDiftObQfhlPOnDxpoEN0tJEGr-Q-6cVNg")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Enable CORS for Flutter Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data = joblib.load("triage_model.pkl")
model = data['model']
maps = data['maps']

class PatientData(BaseModel):
    age: int; pain_level: int; vitals: str; gender: str; insurance: str; blind_audit: bool
class ClinicalData(BaseModel):
    age: int; pain_level: int; vitals: str
class AuditResult(BaseModel):
    gender: str; insurance: str; biased_score: int; fair_score: int; gender_penalty: int; insurance_penalty: int

def get_decision(score):
    if score >= 80: return "Immediate Admission"
    elif score >= 50: return "Standard Queue"
    else: return "Discharge/Waitlist"

@app.post("/predict")
def predict_triage(patient: PatientData):
    vitals_enc = maps['vitals'][patient.vitals]
    gender_enc = maps['gender'][patient.gender]
    insurance_enc = maps['insurance'][patient.insurance]
    spending = 500 if patient.insurance == "State-Funded" else 5000

    input_data = pd.DataFrame([{'age': patient.age, 'pain_level': patient.pain_level, 'vitals': vitals_enc, 'gender': gender_enc, 'insurance': insurance_enc, 'historical_spending': spending}])
    biased_score = int(round(model.predict(input_data)[0]))

    fair_data = input_data.copy()
    fair_data['gender'] = maps['gender']['Male']; fair_data['insurance'] = maps['insurance']['Premium Private']; fair_data['historical_spending'] = 5000
    fair_score = int(round(model.predict(fair_data)[0]))

    gender_test = input_data.copy()
    gender_test['gender'] = maps['gender']['Male']
    gender_penalty = biased_score - int(round(model.predict(gender_test)[0]))

    insurance_test = input_data.copy()
    insurance_test['insurance'] = maps['insurance']['Premium Private']; insurance_test['historical_spending'] = 5000
    insurance_penalty = biased_score - int(round(model.predict(insurance_test)[0]))

    final_score = fair_score if patient.blind_audit else biased_score
    return {"priority_score": final_score, "triage_decision": get_decision(final_score), "feature_attributions": {"gender_penalty": int(gender_penalty) if gender_penalty < 0 else 0, "insurance_penalty": int(insurance_penalty) if insurance_penalty < 0 else 0}}

@app.post("/hunt_bias")
def adversarial_hunt(patient: ClinicalData):
    vitals_enc = maps['vitals'][patient.vitals]
    worst_score = 100; worst_demographics = {}
    for gender in ["Male", "Female"]:
        for insurance in ["Premium Private", "State-Funded"]:
            spending = 500 if insurance == "State-Funded" else 5000
            test_data = pd.DataFrame([{'age': patient.age, 'pain_level': patient.pain_level, 'vitals': vitals_enc, 'gender': maps['gender'][gender], 'insurance': maps['insurance'][insurance], 'historical_spending': spending}])
            score = int(round(model.predict(test_data)[0]))
            if score < worst_score: worst_score = score; worst_demographics = {"gender": gender, "insurance": insurance}

    fair_data = pd.DataFrame([{'age': patient.age, 'pain_level': patient.pain_level, 'vitals': vitals_enc, 'gender': maps['gender']["Male"], 'insurance': maps['insurance']["Premium Private"], 'historical_spending': 5000}])
    fair_score = int(round(model.predict(fair_data)[0]))
    return {"vulnerability_found": True, "worst_case_score": worst_score, "fair_baseline_score": fair_score, "targeted_demographics": worst_demographics, "total_bias_penalty": worst_score - fair_score}

@app.post("/generate_report")
def generate_report(data: AuditResult):
    prompt = f"""
    Act as an expert AI Compliance Officer for a major hospital network. 
    We just audited our medical triage AI and found the following bias:
    - Patient Profile: {data.gender} patient with {data.insurance} insurance.
    - The AI gave them a biased priority score of {data.biased_score}.
    - When we stripped the demographic proxy weights, the fair score was {data.fair_score}.
    - The AI mathematically penalized them {data.gender_penalty} points for their gender and {data.insurance_penalty} points for their insurance status.

    Write a highly visual, striking "Urgent Compliance Alert". Do NOT write a wall of text. 
    Format it exactly like this using Markdown:
    
    🚨 **URGENT COMPLIANCE ALERT: SYSTEMIC AI BIAS DETECTED**
    
    **The Vulnerability**
    [1-2 punchy sentences explaining the exact point penalties and demographic targeting found in the data].
    
    **The Liability**
    [Bulleted list of 3 major risks, explicitly mentioning UN SDG 10 (Reduced Inequalities) and legal exposure].
    
    **Immediate Action Required**
    [2 bullet points outlining the immediate enforcement of the 'Blind Audit' UI and re-engineering the model].
    """
    response = gemini_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
    return {"gemini_report": response.text}
