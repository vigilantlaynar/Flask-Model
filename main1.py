import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


data = pd.read_csv('twitter_sentiments.csv')
print(data.head())
data.label.value_counts()

train, test = train_test_split(data, test_size = 0.2, stratify = data['label'], random_state=21)
print("\n\nTrain Data Shape: ", train.shape)
print("\n\nTest Data Shape", test.shape)

print(train.label.value_counts(normalize=True))
print(test.label.value_counts(normalize=True))


tfidf_vectorizer = TfidfVectorizer(lowercase= True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)
tfidf_vectorizer.fit(train.tweet)
train_idf = tfidf_vectorizer.transform(train.tweet)
test_idf  = tfidf_vectorizer.transform(test.tweet)

model_LR = LogisticRegression()
model_LR.fit(train_idf, train.label)


predict_train = model_LR.predict(train_idf)
predict_test = model_LR.predict(test_idf)

print("\n\nF1 SCORE TRAIN DATA",f1_score(y_true= train.label, y_pred= predict_train))
print("\n\n F1 SCORE TEST DATA",f1_score(y_true= test.label, y_pred= predict_test))

pipeline = Pipeline(steps= [('tfidf', TfidfVectorizer(lowercase=True,max_features=1000,stop_words= ENGLISH_STOP_WORDS)),
                            ('model', LogisticRegression())])


pipeline.fit(train.tweet, train.label)
print(pipeline.predict(train.tweet))


text = ["@user #sikh #temple vandalised in in #calgary, #wso condemns  act "]
print(pipeline.predict(text))

from joblib import dump
dump(pipeline, filename="text_classification.joblib")

var = data[data.label == 1]

