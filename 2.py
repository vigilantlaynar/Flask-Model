import joblib
from joblib import load


text = ["@user #sikh #temple vandalised in in #calgary, #wso condemns  act"]


pipeline = load("text_classification.joblib")
print(pipeline.predict(text))