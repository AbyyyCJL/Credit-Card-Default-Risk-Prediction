import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import classification_report

# Step 1: Load data
df = pd.read_csv("data/Credit_Card.csv")

# Step 2: Drop ID column and apply initial preprocessing
df.drop(columns=["ID"], inplace=True)

# Step 3: Feature Engineering
# Create new features
# 1. Total Bill Amount
df["TOTAL_BILL"] = df[[
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
]].sum(axis=1)

# 2. Total Payment Amount
df["TOTAL_PAYMENT"] = df[[
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]].sum(axis=1)

# 3. Payment Ratio (avoid division by zero)
df["PAYMENT_RATIO"] = df["TOTAL_PAYMENT"] / df["TOTAL_BILL"].replace(0, 1)

# 4. Average Delay
df["AVG_DELAY"] = df[[
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
]].mean(axis=1)

# Step 4: Separate features and target
X = df.drop(columns=["default.payment.next.month"])
y = df["default.payment.next.month"]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Step 6: Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Step 7: Feature Scaling
scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_resampled)

# Save the scaler
joblib.dump(scaler, "models/scaler.pkl")

# Step 8: Define models
models = {
    "xgboost_model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "randomforest_model": RandomForestClassifier(random_state=42),
    "logisticregression_model": LogisticRegression(max_iter=1000)
}

# Step 9: Train and save each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_res_scaled, y_resampled)

    # Evaluate
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    print(f"{name} Report:\n", classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, f"models/{name}.pkl")
    print(f"{name} saved to models/{name}.pkl")
