
TI-One Logistic Regression → COS Upload

1. Replace credentials in train.py

SECRET_ID = "YOUR_SECRET_ID_HERE"
SECRET_KEY = "YOUR_SECRET_KEY_HERE"

Get them from:
Tencent Cloud → CAM → API Keys

2. Startup command

cd /workspace
/usr/local/python3/bin/python3 -m pip install -r requirements.txt
/usr/local/python3/bin/python3 train.py

3. After training

Model will upload to:

COS bucket
lr-1327752399/models/lr_model.pkl
