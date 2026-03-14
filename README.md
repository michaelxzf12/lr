
TI-One Logistic Regression → COS Upload (Environment Variables)

1. Startup command example:

cd /workspace
export COS_SECRET_ID=AKIDxxxxxxxxxxxx
export COS_SECRET_KEY=xxxxxxxxxxxxxxxx

/usr/local/python3/bin/python3 -m pip install -r requirements.txt
/usr/local/python3/bin/python3 train.py

2. After training the model will upload to:

COS bucket:
lr-1327752399/models/lr_model.pkl
