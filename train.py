
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from qcloud_cos import CosConfig, CosS3Client

print("start training...")

data = pd.read_csv("dataset.csv")

X = data.drop("default", axis=1)
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred_proba = model.predict_proba(X_test)[:,1]
pred_label = model.predict(X_test)

print("AUC:", roc_auc_score(y_test, pred_proba))
print(classification_report(y_test, pred_label))

# Save model locally
model_file = "lr_model.pkl"
joblib.dump(model, model_file)
print("Model saved locally:", model_file)

# =============================
# COS Upload Section
# =============================

SECRET_ID = os.environ["COS_SECRET_ID"]
SECRET_KEY = os.environ["COS_SECRET_KEY"]

REGION = "ap-singapore"
BUCKET = "lr-1327752399"

config = CosConfig(
    Region=REGION,
    SecretId=SECRET_ID,
    SecretKey=SECRET_KEY
)

client = CosS3Client(config)

response = client.upload_file(
    Bucket=BUCKET,
    LocalFilePath=model_file,
    Key="models/lr_model.pkl"
)

print("Uploaded model to COS")
print("Bucket:", BUCKET)
print("Path: models/lr_model.pkl")
print("end training...")
