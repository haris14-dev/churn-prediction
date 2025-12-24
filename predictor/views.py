from django.shortcuts import render
from django.conf import settings
import pandas as pd
import joblib
import os

# Absolute paths to ML assets
MODEL_PATH = os.path.join(settings.BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(settings.BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(settings.BASE_DIR, "models", "feature_names.pkl")

# Load trained assets
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)


def home(request):
    prediction = None

    if request.method == "POST":
        # 1️⃣ Collect RAW user inputs
        user_data = {
            "gender": request.POST.get("gender"),
            "SeniorCitizen": int(request.POST.get("SeniorCitizen")),
            "Partner": request.POST.get("Partner"),
            "Dependents": request.POST.get("Dependents"),
            "tenure": int(request.POST.get("tenure")),
            "PhoneService": request.POST.get("PhoneService"),
            "PaperlessBilling": request.POST.get("PaperlessBilling"),
            "MonthlyCharges": float(request.POST.get("MonthlyCharges")),
            "TotalCharges": float(request.POST.get("TotalCharges")),
            "MultipleLines": request.POST.get("MultipleLines"),
            "InternetService": request.POST.get("InternetService"),
            "OnlineSecurity": request.POST.get("OnlineSecurity"),
            "OnlineBackup": request.POST.get("OnlineBackup"),
            "DeviceProtection": request.POST.get("DeviceProtection"),
            "TechSupport": request.POST.get("TechSupport"),
            "StreamingTV": request.POST.get("StreamingTV"),
            "StreamingMovies": request.POST.get("StreamingMovies"),
            "Contract": request.POST.get("Contract"),
            "PaymentMethod": request.POST.get("PaymentMethod"),
        }

        # 2️⃣ Convert to DataFrame
        df = pd.DataFrame([user_data])

        # 3️⃣ Apply SAME encoding as training
        df_encoded = pd.get_dummies(df)

        # 4️⃣ Add missing columns
        for col in feature_names:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        # 5️⃣ Reorder columns EXACTLY
        df_encoded = df_encoded[feature_names]

        # 6️⃣ Scale
        df_scaled = scaler.transform(df_encoded)

        # 7️⃣ Predict
        pred = model.predict(df_scaled)[0]
        prediction = "Churn" if pred == 1 else "No Churn"

    return render(request, "index.html", {"prediction": prediction})
